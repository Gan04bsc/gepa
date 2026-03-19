"""Run PUPA-style GEPA experiments with multi-seed reporting.

This runner aligns the requested setup:
- Task: PUPA-style privacy-conscious delegation
- Model: Qwen3-8B (configurable)
- Optimizer: GEPA, Merge disabled
- Budget: 1200 rollouts
- Minibatch: 3
- Split: train/val/test = 111/60/221
- Seeds: at least 3

It records:
- best val score
- final test score
- mean/std across seeds
- final prompt length
- module update counts
- feedback category for each accepted update
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import gepa
from examples.minimal_quickstart_openai import API_KEY, BASE_URL
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from gepa.core.callbacks import CandidateAcceptedEvent, ProposalStartEvent, ValsetEvaluatedEvent


class PUPADataInst(TypedDict):
    user_query: str
    reference_answer: str
    pii_tokens: list[str]
    additional_context: dict[str, str]


class PUPARolloutOutput(TypedDict):
    redacted_request: str
    untrusted_response: str
    final_response: str


class PUPATrajectory(TypedDict):
    data: PUPADataInst
    redacted_request: str
    untrusted_response: str
    final_response: str
    aggregate_score: float
    objective_scores: dict[str, float]
    feedback_by_module: dict[str, str]


class UpdateTrackerCallback:
    """Tracks accepted component updates and feedback categories."""

    def __init__(self) -> None:
        self._proposal_start_by_iter: dict[int, ProposalStartEvent] = {}
        self.module_update_counts: Counter[str] = Counter()
        self.updates: list[dict[str, Any]] = []
        self.best_val_events: list[dict[str, Any]] = []

    def on_proposal_start(self, event: ProposalStartEvent) -> None:
        self._proposal_start_by_iter[event["iteration"]] = event

    def on_candidate_accepted(self, event: CandidateAcceptedEvent) -> None:
        iteration = event["iteration"]
        proposal_event = self._proposal_start_by_iter.get(iteration)
        if proposal_event is None:
            return
        components = proposal_event["components"]
        reflective_dataset = proposal_event["reflective_dataset"]

        for module_name in components:
            self.module_update_counts[module_name] += 1
            module_records = reflective_dataset.get(module_name, [])
            feedback_categories = sorted(classify_feedback_records(module_records))
            self.updates.append(
                {
                    "iteration": iteration,
                    "module": module_name,
                    "feedback_categories": feedback_categories,
                    "num_reflective_records": len(module_records),
                }
            )

    def on_valset_evaluated(self, event: ValsetEvaluatedEvent) -> None:
        if event["is_best_program"]:
            self.best_val_events.append(
                {
                    "iteration": event["iteration"],
                    "candidate_idx": event["candidate_idx"],
                    "average_score": event["average_score"],
                    "num_examples_evaluated": event["num_examples_evaluated"],
                    "total_valset_size": event["total_valset_size"],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed PUPA GEPA experiment with full metric recording.")
    parser.add_argument("--tag", default="pupa_qwen3_gepa", help="Tag appended to run directory.")
    parser.add_argument(
        "--resume-run-dir",
        type=str,
        default=None,
        help="Resume from an existing root run directory (e.g., gepa_runs/pupa_qwen3_gepa/<timestamp>_<tag>).",
    )
    parser.add_argument("--model", default="openai/qwen3-8b", help="Task/reflection model name for LiteLLM.")
    parser.add_argument("--train-file", type=str, default=None, help="Train split JSON/JSONL path.")
    parser.add_argument("--val-file", type=str, default=None, help="Validation split JSON/JSONL path.")
    parser.add_argument("--test-file", type=str, default=None, help="Test split JSON/JSONL path.")
    parser.add_argument("--all-data-file", type=str, default=None, help="Single JSON/JSONL file to split into train/val/test.")
    parser.add_argument("--split-seed", type=int, default=0, help="Seed used for sampling/splitting dataset.")
    parser.add_argument("--train-size", type=int, default=111, help="Train set size.")
    parser.add_argument("--val-size", type=int, default=60, help="Validation set size.")
    parser.add_argument("--test-size", type=int, default=221, help="Test set size.")
    parser.add_argument("--seeds", type=str, default="11,22,33", help="Comma-separated GEPA seeds (1+; 3+ recommended).")
    parser.add_argument("--max-metric-calls", type=int, default=1200, help="GEPA rollout budget.")
    parser.add_argument("--reflection-minibatch-size", type=int, default=3, help="Reflection minibatch size.")
    parser.add_argument("--quality-weight", type=float, default=0.7, help="Weight for quality score.")
    parser.add_argument("--privacy-weight", type=float, default=0.3, help="Weight for privacy score.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Task LM temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Task LM top-p.")
    parser.add_argument("--top-k", type=int, default=20, help="Task LM top-k (set <0 to disable).")
    parser.add_argument("--reflection-temperature", type=float, default=0.6, help="Reflection LM temperature.")
    parser.add_argument("--reflection-top-p", type=float, default=0.95, help="Reflection LM top-p.")
    parser.add_argument("--reflection-top-k", type=int, default=20, help="Reflection LM top-k (set <0 to disable).")
    parser.add_argument("--max-retries", type=int, default=20, help="Max retries for each model call.")
    parser.add_argument("--initial-backoff-seconds", type=float, default=2.0, help="Initial retry backoff.")
    parser.add_argument("--max-backoff-seconds", type=float, default=20.0, help="Retry backoff cap.")
    parser.add_argument("--timeout-seconds", type=float, default=120.0, help="Timeout per call.")
    parser.add_argument("--min-call-interval-seconds", type=float, default=0.8, help="Throttle between calls.")
    parser.add_argument(
        "--thinking-mode",
        type=str,
        choices=["auto", "off", "on"],
        default="auto",
        help="Control provider-specific thinking flag. Use 'off' for gateways requiring enable_thinking=false.",
    )
    parser.add_argument("--disable-progress-bar", action="store_true", help="Disable GEPA tqdm progress bar.")
    parser.add_argument("--dry-run", action="store_true", help="Run with a local mock LM (no API calls).")
    return parser.parse_args()


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        items: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
        return items

    loaded = json.loads(text)
    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, dict):
        for key in ["data", "items", "examples", "records"]:
            value = loaded.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported JSON schema in {path}")


def pick_first_str(d: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = d.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extract_pii_tokens(raw: dict[str, Any], user_query: str) -> list[str]:
    for key in ["pii_tokens", "pii", "sensitive_terms", "private_entities", "sensitive_spans"]:
        value = raw.get(key)
        if isinstance(value, list):
            tokens = [str(v).strip() for v in value if str(v).strip()]
            if tokens:
                return tokens
        if isinstance(value, str) and value.strip():
            pieces = [p.strip() for p in re.split(r"[,\n;|]", value) if p.strip()]
            if pieces:
                return pieces

    email_like = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", user_query)
    phone_like = re.findall(r"\b\d{11}\b|\b\d{3}[- ]?\d{4}[- ]?\d{4}\b", user_query)
    id_like = re.findall(r"\b\d{15,18}[xX]?\b", user_query)
    return sorted(set(email_like + phone_like + id_like))


def normalize_item(raw: dict[str, Any]) -> PUPADataInst:
    user_query = pick_first_str(raw, ["user_query", "query", "input", "private_query", "original_query"])
    if user_query is None:
        raise ValueError(f"Cannot find user query in record keys={sorted(raw.keys())}")

    reference_answer = pick_first_str(
        raw,
        ["reference_answer", "reference_response", "answer", "target_answer", "target_response", "response"],
    )
    if reference_answer is None:
        reference_answer = ""

    pii_tokens = extract_pii_tokens(raw, user_query)
    additional_context = raw.get("additional_context")
    if not isinstance(additional_context, dict):
        additional_context = {}

    return {
        "user_query": user_query,
        "reference_answer": reference_answer,
        "pii_tokens": pii_tokens,
        "additional_context": {str(k): str(v) for k, v in additional_context.items()},
    }


def sample_exact(items: list[PUPADataInst], target_size: int, rng: random.Random, split_name: str) -> list[PUPADataInst]:
    if len(items) < target_size:
        raise ValueError(f"{split_name} has {len(items)} items, but requested {target_size}.")
    if len(items) == target_size:
        return list(items)
    sampled_indices = sorted(rng.sample(range(len(items)), target_size))
    return [items[i] for i in sampled_indices]


def load_and_prepare_dataset(args: argparse.Namespace) -> tuple[list[PUPADataInst], list[PUPADataInst], list[PUPADataInst]]:
    rng = random.Random(args.split_seed)

    if args.dry_run:
        data = build_dry_run_dataset(max(args.train_size, 12) + max(args.val_size, 8) + max(args.test_size, 8))
        rng.shuffle(data)
        train = sample_exact(data, args.train_size, rng, "train")
        remaining = [x for x in data if x not in train]
        val = sample_exact(remaining, args.val_size, rng, "val")
        remaining = [x for x in remaining if x not in val]
        test = sample_exact(remaining, args.test_size, rng, "test")
        return train, val, test

    if args.all_data_file:
        all_items = [normalize_item(item) for item in read_json_or_jsonl(Path(args.all_data_file))]
        if len(all_items) < args.train_size + args.val_size + args.test_size:
            raise ValueError(
                f"all-data-file has only {len(all_items)} rows, "
                f"but needs at least {args.train_size + args.val_size + args.test_size}."
            )
        rng.shuffle(all_items)
        train = all_items[: args.train_size]
        val = all_items[args.train_size : args.train_size + args.val_size]
        test = all_items[args.train_size + args.val_size : args.train_size + args.val_size + args.test_size]
        return train, val, test

    if not (args.train_file and args.val_file and args.test_file):
        raise ValueError("Provide either --all-data-file, or all of --train-file/--val-file/--test-file.")

    train_raw = [normalize_item(item) for item in read_json_or_jsonl(Path(args.train_file))]
    val_raw = [normalize_item(item) for item in read_json_or_jsonl(Path(args.val_file))]
    test_raw = [normalize_item(item) for item in read_json_or_jsonl(Path(args.test_file))]
    train = sample_exact(train_raw, args.train_size, rng, "train")
    val = sample_exact(val_raw, args.val_size, rng, "val")
    test = sample_exact(test_raw, args.test_size, rng, "test")
    return train, val, test


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def f1_token_overlap(pred: str, gold: str) -> float:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    if not gold_tokens:
        return 0.0 if pred_tokens else 1.0
    if not pred_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum(min(pred_counts[t], gold_counts[t]) for t in gold_counts)
    precision = overlap / max(1, len(pred_tokens))
    recall = overlap / max(1, len(gold_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def leakage_score(text: str, pii_tokens: list[str]) -> float:
    if not pii_tokens:
        return 1.0
    text_lower = text.lower()
    hits = 0
    for token in pii_tokens:
        t = token.strip().lower()
        if not t:
            continue
        if t in text_lower:
            hits += 1
    return 1.0 - min(1.0, hits / max(1, len(pii_tokens)))


def classify_feedback_records(records: list[dict[str, Any]]) -> set[str]:
    full_text = []
    for record in records:
        feedback = record.get("Feedback")
        if isinstance(feedback, str):
            full_text.append(feedback.lower())
        else:
            full_text.append(json.dumps(feedback, ensure_ascii=False).lower())
    joined = "\n".join(full_text)

    categories = set()
    if any(k in joined for k in ["pii", "privacy", "leak", "redact", "anonym"]):
        categories.add("privacy_or_pii")
    if any(k in joined for k in ["quality", "correct", "answer", "relevance", "helpful", "coverage"]):
        categories.add("response_quality")
    if any(k in joined for k in ["format", "constraint", "structure", "json", "style"]):
        categories.add("format_or_constraint")
    if any(k in joined for k in ["hallucination", "fabricat", "unsupported"]):
        categories.add("factuality_risk")
    if not categories:
        categories.add("other")
    return categories


def make_retry_lm_callable(
    model_name: str,
    base_url: str,
    api_key: str,
    max_retries: int,
    initial_backoff_seconds: float,
    max_backoff_seconds: float,
    timeout_seconds: float,
    min_call_interval_seconds: float,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    enable_thinking: bool | None,
):
    import litellm

    last_call_ts = 0.0

    def _is_retryable_exception(exc: Exception) -> bool:
        text = f"{type(exc).__name__} {exc!r} {exc!s}".lower()
        retry_tokens = (
            "ratelimit",
            "rate limit",
            "too_many_requests",
            "429",
            "timeout",
            "timed out",
            "temporarily",
            "overloaded",
            "connection",
            "connecterror",
            "service unavailable",
            "503",
        )
        return any(token in text for token in retry_tokens)

    def _call(prompt: str | list[dict[str, str]]) -> str:
        nonlocal last_call_ts
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        last_exception: Exception | None = None
        for attempt in range(max_retries):
            try:
                now = time.time()
                delta = now - last_call_ts
                if delta < min_call_interval_seconds:
                    time.sleep(min_call_interval_seconds - delta)

                request_kwargs: dict[str, Any] = {
                    "model": model_name,
                    "messages": messages,
                    "api_key": api_key,
                    "base_url": base_url,
                    "timeout": timeout_seconds,
                }
                if temperature is not None:
                    request_kwargs["temperature"] = temperature
                if top_p is not None:
                    request_kwargs["top_p"] = top_p
                if top_k is not None and top_k >= 0:
                    request_kwargs["top_k"] = top_k
                if enable_thinking is not None:
                    request_kwargs["extra_body"] = {"enable_thinking": enable_thinking}

                response = litellm.completion(**request_kwargs)
                last_call_ts = time.time()
                content = response.choices[0].message.content
                return content.strip() if isinstance(content, str) else str(content)
            except Exception as exc:
                last_exception = exc
                is_retryable = _is_retryable_exception(exc)
                if attempt == max_retries - 1 or not is_retryable:
                    raise
                base_sleep = min(max_backoff_seconds, initial_backoff_seconds * (2**attempt))
                sleep_seconds = base_sleep + random.uniform(0.0, 0.3)
                print(
                    f"[LM retry] model={model_name} attempt={attempt + 1}/{max_retries} "
                    f"sleep={sleep_seconds:.2f}s error={type(exc).__name__}"
                )
                time.sleep(sleep_seconds)

        raise RuntimeError(f"Model call failed after {max_retries} retries: {last_exception}")

    return _call


def make_mock_lm_callable():
    def _mock(prompt: str | list[dict[str, str]]) -> str:
        if isinstance(prompt, str):
            lower_prompt = prompt.lower()
            if "component" in lower_prompt and "instruction" in lower_prompt:
                return (
                    "Rewrite private user requests into abstract task descriptions, "
                    "remove direct identifiers, and preserve user intent and constraints."
                )
            return "Improve clarity, keep constraints explicit, and avoid private identifiers."

        user_text = prompt[-1]["content"] if prompt else ""
        lower = user_text.lower()
        if "private user query" in lower or "redacted request" in lower:
            redacted = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL]", user_text)
            redacted = re.sub(r"\b\d{11,18}[xX]?\b", "[ID]", redacted)
            redacted = re.sub(r"\b\d{3}[- ]?\d{4}[- ]?\d{4}\b", "[PHONE]", redacted)
            return f"Please answer the following request safely: {redacted[:300]}"
        if "assistant for generic help" in lower:
            return "General advice: prioritize clear steps and avoid exposing private details."
        return "Here is a privacy-preserving response with practical steps and no private identifiers."

    return _mock


def resolve_enable_thinking(model_name: str, thinking_mode: str) -> bool | None:
    if thinking_mode == "off":
        return False
    if thinking_mode == "on":
        return True
    lower = model_name.lower()
    if "qwen" in lower:
        return False
    return None


class PUPAAdapter(GEPAAdapter[PUPADataInst, PUPATrajectory, PUPARolloutOutput]):
    """A minimal two-module PUPA-style adapter.

    Components optimized by GEPA:
    - craft_redacted_request
    - respond_to_query
    """

    def __init__(
        self,
        model_callable,
        quality_weight: float = 0.7,
        privacy_weight: float = 0.3,
    ) -> None:
        self.model_callable = model_callable
        self.quality_weight = quality_weight
        self.privacy_weight = privacy_weight

    def evaluate(
        self,
        batch: list[PUPADataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[PUPATrajectory, PUPARolloutOutput]:
        outputs: list[PUPARolloutOutput] = []
        scores: list[float] = []
        objective_scores: list[dict[str, float]] = []
        trajectories: list[PUPATrajectory] | None = [] if capture_traces else None

        craft_prompt = candidate["craft_redacted_request"]
        respond_prompt = candidate["respond_to_query"]

        for data in batch:
            user_query = data["user_query"]
            reference_answer = data["reference_answer"]
            pii_tokens = data["pii_tokens"]

            redacted_request = self.model_callable(
                [
                    {"role": "system", "content": craft_prompt},
                    {
                        "role": "user",
                        "content": (
                            "Private user query:\n"
                            f"{user_query}\n\n"
                            "Rewrite it into a privacy-preserving request for an external assistant."
                        ),
                    },
                ]
            )

            untrusted_response = self.model_callable(
                [
                    {"role": "system", "content": "You are an assistant for generic help requests."},
                    {"role": "user", "content": redacted_request},
                ]
            )

            final_response = self.model_callable(
                [
                    {"role": "system", "content": respond_prompt},
                    {
                        "role": "user",
                        "content": (
                            "Original user query:\n"
                            f"{user_query}\n\n"
                            "Redacted request sent to external model:\n"
                            f"{redacted_request}\n\n"
                            "External model response:\n"
                            f"{untrusted_response}\n\n"
                            "Provide the final user-facing response."
                        ),
                    },
                ]
            )

            quality = f1_token_overlap(final_response, reference_answer)
            privacy_redacted = leakage_score(redacted_request, pii_tokens)
            privacy_final = leakage_score(final_response, pii_tokens)
            privacy = min(privacy_redacted, privacy_final)
            aggregate = self.quality_weight * quality + self.privacy_weight * privacy

            craft_feedback = (
                f"privacy_redacted={privacy_redacted:.3f}; "
                f"pii_tokens_count={len(pii_tokens)}; "
                "Improve abstraction/anonymization while preserving user intent."
            )
            respond_feedback = (
                f"quality={quality:.3f}; privacy_final={privacy_final:.3f}; aggregate={aggregate:.3f}; "
                "Improve answer quality and keep privacy leakage at zero."
            )

            output: PUPARolloutOutput = {
                "redacted_request": redacted_request,
                "untrusted_response": untrusted_response,
                "final_response": final_response,
            }
            outputs.append(output)
            scores.append(aggregate)
            objective_scores.append({"quality": quality, "privacy": privacy})

            if trajectories is not None:
                trajectories.append(
                    {
                        "data": data,
                        "redacted_request": redacted_request,
                        "untrusted_response": untrusted_response,
                        "final_response": final_response,
                        "aggregate_score": aggregate,
                        "objective_scores": {"quality": quality, "privacy": privacy},
                        "feedback_by_module": {
                            "craft_redacted_request": craft_feedback,
                            "respond_to_query": respond_feedback,
                        },
                    }
                )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[PUPATrajectory, PUPARolloutOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        if eval_batch.trajectories is None:
            raise ValueError("Trajectories are required for reflective dataset construction.")

        reflective_dataset: dict[str, list[dict[str, Any]]] = {}
        for component in components_to_update:
            rows: list[dict[str, Any]] = []
            for traj in eval_batch.trajectories:
                if component == "craft_redacted_request":
                    rows.append(
                        {
                            "Inputs": {
                                "user_query": traj["data"]["user_query"],
                                "pii_tokens": traj["data"]["pii_tokens"],
                            },
                            "Generated Outputs": traj["redacted_request"],
                            "Feedback": traj["feedback_by_module"]["craft_redacted_request"],
                        }
                    )
                elif component == "respond_to_query":
                    rows.append(
                        {
                            "Inputs": {
                                "user_query": traj["data"]["user_query"],
                                "redacted_request": traj["redacted_request"],
                                "untrusted_response": traj["untrusted_response"],
                                "reference_answer": traj["data"]["reference_answer"],
                            },
                            "Generated Outputs": traj["final_response"],
                            "Feedback": traj["feedback_by_module"]["respond_to_query"],
                        }
                    )
                else:
                    raise ValueError(f"Unknown component: {component}")
            reflective_dataset[component] = rows
        return reflective_dataset


def evaluate_candidate_on_dataset(
    adapter: PUPAAdapter,
    dataset: list[PUPADataInst],
    candidate: dict[str, str],
) -> dict[str, float]:
    eval_result = adapter.evaluate(dataset, candidate, capture_traces=False)
    avg_score = sum(eval_result.scores) / max(1, len(eval_result.scores))
    quality_scores = []
    privacy_scores = []
    if eval_result.objective_scores is not None:
        for obj in eval_result.objective_scores:
            quality_scores.append(obj.get("quality", 0.0))
            privacy_scores.append(obj.get("privacy", 0.0))
    return {
        "aggregate": avg_score,
        "quality": sum(quality_scores) / max(1, len(quality_scores)),
        "privacy": sum(privacy_scores) / max(1, len(privacy_scores)),
    }


def parse_seed_list(seed_str: str) -> list[int]:
    seeds = [int(s.strip()) for s in seed_str.split(",") if s.strip()]
    if len(seeds) < 1:
        raise ValueError("Please provide at least 1 seed, e.g. --seeds 11")
    return seeds


def prompt_length_stats(candidate: dict[str, str]) -> dict[str, Any]:
    per_module = {}
    total_chars = 0
    total_words = 0
    for module, text in candidate.items():
        char_len = len(text)
        word_len = len(tokenize(text))
        per_module[module] = {"chars": char_len, "words": word_len}
        total_chars += char_len
        total_words += word_len
    return {"total_chars": total_chars, "total_words": total_words, "per_module": per_module}


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def build_seed_candidate() -> dict[str, str]:
    return {
        "craft_redacted_request": (
            "You rewrite private user queries into privacy-preserving external requests. "
            "Remove direct identifiers (names, phone, email, id numbers, exact address) while keeping intent."
        ),
        "respond_to_query": (
            "You produce the final user-facing answer based on the original query and external response. "
            "Keep the answer useful, concise, and avoid revealing any private identifiers."
        ),
    }


def build_dry_run_dataset(n: int) -> list[PUPADataInst]:
    data: list[PUPADataInst] = []
    for i in range(n):
        email = f"user{i}@example.com"
        phone = f"1380000{i:04d}"
        query = (
            f"My name is User{i}, email {email}, phone {phone}. "
            "Please suggest a weekly plan to improve sleep and focus."
        )
        data.append(
            {
                "user_query": query,
                "reference_answer": "Provide weekly sleep and focus plan with clear daily steps.",
                "pii_tokens": [email, phone, f"User{i}"],
                "additional_context": {},
            }
        )
    return data


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    if len(seeds) < 3:
        print(f"[Warning] only {len(seeds)} seed(s) provided: {seeds}. This is fine for a quick run but less stable.")

    if not args.dry_run and API_KEY == "YOUR_API_KEY_HERE":
        raise SystemExit("Please set API_KEY in examples/minimal_quickstart_openai.py or run with --dry-run.")

    trainset, valset, testset = load_and_prepare_dataset(args)
    if len(trainset) != args.train_size or len(valset) != args.val_size or len(testset) != args.test_size:
        raise SystemExit("Prepared split size does not match requested train/val/test sizes.")

    if args.resume_run_dir:
        root_run_dir = Path(args.resume_run_dir)
        root_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Config] resume_run_dir={root_run_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_run_dir = Path("gepa_runs") / "pupa_qwen3_gepa" / f"{timestamp}_{args.tag}"
        root_run_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest = {
        "train_size": len(trainset),
        "val_size": len(valset),
        "test_size": len(testset),
        "split_seed": args.split_seed,
        "train_file": args.train_file,
        "val_file": args.val_file,
        "test_file": args.test_file,
        "all_data_file": args.all_data_file,
        "dry_run": args.dry_run,
    }
    (root_run_dir / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, ensure_ascii=False, indent=2))

    print(
        f"[Config] model={args.model} budget={args.max_metric_calls} minibatch={args.reflection_minibatch_size} "
        f"split={len(trainset)}/{len(valset)}/{len(testset)} seeds={seeds}"
    )

    resolved_enable_thinking: bool | None = None
    if args.dry_run:
        task_lm_callable = make_mock_lm_callable()
        reflection_lm_callable = make_mock_lm_callable()
    else:
        resolved_enable_thinking = resolve_enable_thinking(args.model, args.thinking_mode)
        print(f"[Config] thinking_mode={args.thinking_mode} resolved_enable_thinking={resolved_enable_thinking}")
        task_top_k = args.top_k if args.top_k >= 0 else None
        reflection_top_k = args.reflection_top_k if args.reflection_top_k >= 0 else None
        task_lm_callable = make_retry_lm_callable(
            model_name=args.model,
            base_url=BASE_URL,
            api_key=API_KEY,
            max_retries=args.max_retries,
            initial_backoff_seconds=args.initial_backoff_seconds,
            max_backoff_seconds=args.max_backoff_seconds,
            timeout_seconds=args.timeout_seconds,
            min_call_interval_seconds=args.min_call_interval_seconds,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=task_top_k,
            enable_thinking=resolved_enable_thinking,
        )
        reflection_lm_callable = make_retry_lm_callable(
            model_name=args.model,
            base_url=BASE_URL,
            api_key=API_KEY,
            max_retries=args.max_retries,
            initial_backoff_seconds=args.initial_backoff_seconds,
            max_backoff_seconds=args.max_backoff_seconds,
            timeout_seconds=args.timeout_seconds,
            min_call_interval_seconds=args.min_call_interval_seconds,
            temperature=args.reflection_temperature,
            top_p=args.reflection_top_p,
            top_k=reflection_top_k,
            enable_thinking=resolved_enable_thinking,
        )

    seed_summaries: list[dict[str, Any]] = []

    for seed in seeds:
        print(f"[Seed {seed}] starting...")
        adapter = PUPAAdapter(
            model_callable=task_lm_callable,
            quality_weight=args.quality_weight,
            privacy_weight=args.privacy_weight,
        )
        callback = UpdateTrackerCallback()

        run_dir = root_run_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = gepa.optimize(
            seed_candidate=build_seed_candidate(),
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            task_lm=None,
            reflection_lm=reflection_lm_callable,
            use_merge=False,
            max_metric_calls=args.max_metric_calls,
            reflection_minibatch_size=args.reflection_minibatch_size,
            seed=seed,
            callbacks=[callback],
            run_dir=str(run_dir),
            display_progress_bar=not args.disable_progress_bar,
        )

        best_candidate = result.best_candidate
        if not isinstance(best_candidate, dict):
            raise ValueError("Expected dict candidate for multi-module PUPA experiment.")

        best_val_score = result.val_aggregate_scores[result.best_idx]
        test_metrics = evaluate_candidate_on_dataset(adapter, testset, best_candidate)
        prompt_lengths = prompt_length_stats(best_candidate)

        update_rows = callback.updates
        feedback_category_counts: Counter[str] = Counter()
        for row in update_rows:
            for cat in row["feedback_categories"]:
                feedback_category_counts[cat] += 1

        seed_summary = {
            "seed": seed,
            "run_dir": str(run_dir),
            "best_candidate_idx": result.best_idx,
            "num_candidates": result.num_candidates,
            "total_metric_calls": result.total_metric_calls,
            "best_val_score": best_val_score,
            "final_test_score": test_metrics["aggregate"],
            "final_test_quality_score": test_metrics["quality"],
            "final_test_privacy_score": test_metrics["privacy"],
            "final_prompt_length": prompt_lengths,
            "module_update_counts": dict(callback.module_update_counts),
            "feedback_category_counts": dict(feedback_category_counts),
            "accepted_updates": update_rows,
            "best_val_events": callback.best_val_events,
            "best_candidate": best_candidate,
        }
        seed_summaries.append(seed_summary)

        (run_dir / "seed_summary.json").write_text(
            json.dumps(seed_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "accepted_updates.json").write_text(
            json.dumps(update_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[Seed {seed}] done: best_val={best_val_score:.4f} "
            f"final_test={test_metrics['aggregate']:.4f} updates={len(update_rows)}"
        )

    best_val_scores = [float(s["best_val_score"]) for s in seed_summaries]
    final_test_scores = [float(s["final_test_score"]) for s in seed_summaries]
    prompt_chars = [float(s["final_prompt_length"]["total_chars"]) for s in seed_summaries]

    module_updates_all: dict[str, list[int]] = defaultdict(list)
    for summary in seed_summaries:
        for module_name in ["craft_redacted_request", "respond_to_query"]:
            module_updates_all[module_name].append(int(summary["module_update_counts"].get(module_name, 0)))

    module_update_stats = {module: mean_std(values) for module, values in module_updates_all.items()}

    aggregate_feedback_counts: Counter[str] = Counter()
    for summary in seed_summaries:
        aggregate_feedback_counts.update(summary["feedback_category_counts"])

    aggregate = {
        "config": {
            "task": "PUPA",
            "model": args.model,
            "optimizer": "GEPA",
            "use_merge": False,
            "max_metric_calls": args.max_metric_calls,
            "reflection_minibatch_size": args.reflection_minibatch_size,
            "split": {"train": len(trainset), "val": len(valset), "test": len(testset)},
            "seeds": seeds,
            "quality_weight": args.quality_weight,
            "privacy_weight": args.privacy_weight,
            "thinking_mode": args.thinking_mode,
            "resolved_enable_thinking": resolved_enable_thinking,
            "dry_run": args.dry_run,
            "run_root": str(root_run_dir),
        },
        "best_val_score": {
            "per_seed": best_val_scores,
            **mean_std(best_val_scores),
        },
        "final_test_score": {
            "per_seed": final_test_scores,
            **mean_std(final_test_scores),
        },
        "final_prompt_length_chars": {
            "per_seed": prompt_chars,
            **mean_std(prompt_chars),
        },
        "module_update_stats": module_update_stats,
        "feedback_category_counts": dict(aggregate_feedback_counts),
        "seed_summaries": seed_summaries,
    }

    aggregate_json_path = root_run_dir / "aggregate_summary.json"
    aggregate_json_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# PUPA GEPA Experiment Summary",
        "",
        "## Config",
        f"- Task: `PUPA`",
        f"- Model: `{args.model}`",
        f"- Optimizer: `GEPA` (Merge disabled)",
        f"- Budget: `{args.max_metric_calls}` rollouts",
        f"- Reflection minibatch: `{args.reflection_minibatch_size}`",
        f"- Split: `{len(trainset)} / {len(valset)} / {len(testset)}`",
        f"- Seeds: `{seeds}`",
        "",
        "## Requested Metrics",
        f"- Best val score (mean +/- std): `{aggregate['best_val_score']['mean']:.4f} +/- {aggregate['best_val_score']['std']:.4f}`",
        f"- Final test score (mean +/- std): `{aggregate['final_test_score']['mean']:.4f} +/- {aggregate['final_test_score']['std']:.4f}`",
        (
            "- Final prompt length chars (mean +/- std): "
            f"`{aggregate['final_prompt_length_chars']['mean']:.1f} +/- {aggregate['final_prompt_length_chars']['std']:.1f}`"
        ),
        "",
        "## Module Updates",
        (
            "- craft_redacted_request updates (mean +/- std): "
            f"`{aggregate['module_update_stats'].get('craft_redacted_request', {'mean': 0.0})['mean']:.2f} +/- "
            f"{aggregate['module_update_stats'].get('craft_redacted_request', {'std': 0.0})['std']:.2f}`"
        ),
        (
            "- respond_to_query updates (mean +/- std): "
            f"`{aggregate['module_update_stats'].get('respond_to_query', {'mean': 0.0})['mean']:.2f} +/- "
            f"{aggregate['module_update_stats'].get('respond_to_query', {'std': 0.0})['std']:.2f}`"
        ),
        "",
        "## Feedback Categories (accepted updates)",
    ]
    for key, value in sorted(aggregate_feedback_counts.items()):
        md_lines.append(f"- {key}: `{value}`")

    md_lines.extend(
        [
            "",
            "## Artifacts",
            f"- Aggregate JSON: `{aggregate_json_path}`",
            f"- Per-seed summary: `{root_run_dir}/seed_*/seed_summary.json`",
            f"- Accepted updates: `{root_run_dir}/seed_*/accepted_updates.json`",
        ]
    )

    aggregate_md_path = root_run_dir / "aggregate_summary.md"
    aggregate_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print("[Done] aggregate summary written:")
    print(f"- {aggregate_json_path}")
    print(f"- {aggregate_md_path}")


if __name__ == "__main__":
    main()
