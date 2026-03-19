"""Run one GEPA round on official AIME datasets and persist artifacts.

This uses GEPA's built-in official dataset loader:
    gepa.examples.aime.init_dataset()

It defaults to small sampled subsets to control API cost.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import gepa
from examples.minimal_quickstart_openai import API_KEY, BASE_URL, MODEL_NAME
from gepa.adapters.default_adapter.default_adapter import DefaultDataInst, EvaluationResult
from gepa.examples.aime import init_dataset


class AIMEEvaluator:
    """Evaluator with stronger feedback than default contains-answer check."""

    def __call__(self, data: DefaultDataInst, response: str) -> EvaluationResult:
        answer = data["answer"].strip()
        response_text = response.strip()

        is_correct = answer in response_text
        has_required_format = "### " in response_text
        score = 1.0 if is_correct else 0.0

        if is_correct and has_required_format:
            feedback = f"Correct. Response includes expected final answer {answer} with ### format."
        elif is_correct:
            feedback = (
                f"Partially good. Correct answer {answer} appears, but required format '### <answer>' is missing."
            )
        else:
            feedback = (
                f"Incorrect. Expected final answer {answer}. "
                "Solve carefully and end with exact format: ### <answer>."
            )

        return EvaluationResult(score=score, feedback=feedback, objective_scores=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one GEPA round on official AIME datasets.")
    parser.add_argument("--tag", default="aime_official", help="Tag for run directory.")
    parser.add_argument("--train-size", type=int, default=20, help="Sampled train size from official train split.")
    parser.add_argument("--val-size", type=int, default=20, help="Sampled val size from official val split.")
    parser.add_argument("--dataset-seed", type=int, default=7, help="Sampling seed for reproducibility.")
    parser.add_argument("--max-metric-calls", type=int, default=20, help="Optimization budget.")
    parser.add_argument("--reflection-minibatch-size", type=int, default=1, help="Minibatch size.")
    parser.add_argument("--max-retries", type=int, default=12, help="Max retries per API call.")
    parser.add_argument("--initial-backoff-seconds", type=float, default=1.5, help="Initial retry backoff.")
    parser.add_argument("--max-backoff-seconds", type=float, default=8.0, help="Max retry backoff.")
    parser.add_argument("--timeout-seconds", type=float, default=40.0, help="Per-call timeout.")
    parser.add_argument("--min-call-interval-seconds", type=float, default=2.0, help="Throttle interval.")
    return parser.parse_args()


def make_retry_lm_callable(
    model_name: str,
    base_url: str,
    api_key: str,
    max_retries: int,
    initial_backoff_seconds: float,
    max_backoff_seconds: float,
    timeout_seconds: float,
    min_call_interval_seconds: float,
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
        messages: list[dict[str, str]]
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
                response = litellm.completion(
                    model=model_name,
                    messages=messages,
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout_seconds,
                )
                last_call_ts = time.time()
                content = response.choices[0].message.content
                return content.strip() if isinstance(content, str) else str(content)
            except Exception as exc:
                last_exception = exc
                if attempt == max_retries - 1 or not _is_retryable_exception(exc):
                    raise
                sleep_seconds = min(max_backoff_seconds, initial_backoff_seconds * (2**attempt)) + random.uniform(
                    0.0, 0.3
                )
                print(
                    f"[LM retry] model={model_name} attempt={attempt + 1}/{max_retries} "
                    f"sleep={sleep_seconds:.2f}s error={type(exc).__name__}"
                )
                time.sleep(sleep_seconds)

        raise RuntimeError(f"Model call failed after {max_retries} retries: {last_exception}")

    return _call


def main() -> None:
    args = parse_args()
    if API_KEY == "YOUR_API_KEY_HERE":
        raise SystemExit("Please set API_KEY in examples/minimal_quickstart_openai.py first.")

    os.environ["OPENAI_API_KEY"] = API_KEY
    os.environ["OPENAI_API_BASE"] = BASE_URL

    trainset_full, valset_full, _ = init_dataset()
    rng = random.Random(args.dataset_seed)
    trainset = rng.sample(trainset_full, min(args.train_size, len(trainset_full)))
    valset = rng.sample(valset_full, min(args.val_size, len(valset_full)))

    seed_candidate = {
        "system_prompt": (
            "You are a math assistant. Solve the problem and provide your final answer at the end."
        )
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("gepa_runs") / "one_round_aime_official" / f"{timestamp}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Run config] dataset=official_aime train={len(trainset)} val={len(valset)} seed={args.dataset_seed}")
    print(f"[Run config] seed_prompt={seed_candidate['system_prompt']}")

    lm_callable = make_retry_lm_callable(
        model_name=MODEL_NAME,
        base_url=BASE_URL,
        api_key=API_KEY,
        max_retries=args.max_retries,
        initial_backoff_seconds=args.initial_backoff_seconds,
        max_backoff_seconds=args.max_backoff_seconds,
        timeout_seconds=args.timeout_seconds,
        min_call_interval_seconds=args.min_call_interval_seconds,
    )

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        task_lm=lm_callable,
        reflection_lm=lm_callable,
        evaluator=AIMEEvaluator(),
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        run_dir=str(run_dir),
        display_progress_bar=True,
    )

    seed_score = result.val_aggregate_scores[0]
    best_score = result.val_aggregate_scores[result.best_idx]
    summary = {
        "run_dir": str(run_dir),
        "dataset": "official_aime",
        "train_size": len(trainset),
        "val_size": len(valset),
        "dataset_seed": args.dataset_seed,
        "task_lm": MODEL_NAME,
        "reflection_lm": MODEL_NAME,
        "seed_candidate": seed_candidate,
        "max_metric_calls": args.max_metric_calls,
        "seed_score": seed_score,
        "best_score": best_score,
        "improvement": best_score - seed_score,
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "best_candidate": result.best_candidate,
        "val_aggregate_scores": result.val_aggregate_scores,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

