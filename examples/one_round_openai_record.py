"""Run one full GEPA optimization round and persist artifacts.

This script does NOT hardcode your key again. It reuses the config from:
    examples/minimal_quickstart_openai.py

Run (after filling API_KEY in minimal_quickstart_openai.py):
    python -m examples.one_round_openai_record
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
from examples.minimal_quickstart_openai import API_KEY, BASE_URL, MODEL_NAME, ExactMatchEvaluator
from gepa.adapters.default_adapter.default_adapter import DefaultDataInst


def build_dataset(profile: str) -> list[DefaultDataInst]:
    full = [
        {"input": "What is 7 + 5? Reply with only the number.", "answer": "12", "additional_context": {}},
        {"input": "What is 9 - 4? Reply with only the number.", "answer": "5", "additional_context": {}},
        {"input": "What is 3 * 6? Reply with only the number.", "answer": "18", "additional_context": {}},
        {"input": "What is the capital of France? Reply with one word.", "answer": "Paris", "additional_context": {}},
    ]
    if profile == "smoke":
        return full[:1]
    return full


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one full GEPA optimization round and record results.")
    parser.add_argument("--tag", default="default", help="Tag appended to run folder name.")
    parser.add_argument("--max-metric-calls", type=int, default=30, help="Optimization budget.")
    parser.add_argument("--reflection-minibatch-size", type=int, default=2, help="Minibatch size per reflection step.")
    parser.add_argument("--max-retries", type=int, default=8, help="Max retries for each model call.")
    parser.add_argument("--initial-backoff-seconds", type=float, default=1.0, help="Initial backoff seconds.")
    parser.add_argument("--max-backoff-seconds", type=float, default=12.0, help="Max backoff cap for retries.")
    parser.add_argument("--timeout-seconds", type=float, default=120.0, help="Timeout per model call.")
    parser.add_argument("--min-call-interval-seconds", type=float, default=0.8, help="Throttle interval between LM calls.")
    parser.add_argument(
        "--dataset-profile",
        choices=["smoke", "default"],
        default="default",
        help="Use 'smoke' (1 sample) first when endpoint is unstable.",
    )
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


def main() -> None:
    args = parse_args()

    if API_KEY == "YOUR_API_KEY_HERE":
        raise SystemExit("Please set API_KEY in examples/minimal_quickstart_openai.py first.")

    os.environ["OPENAI_API_KEY"] = API_KEY
    os.environ["OPENAI_API_BASE"] = BASE_URL

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("gepa_runs") / "one_round_openai" / f"{timestamp}_{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    trainset = build_dataset(args.dataset_profile)
    seed_candidate = {
        "system_prompt": "You are a helpful assistant. Answer the user query.",
    }
    print(f"[Run config] dataset_profile={args.dataset_profile} dataset_size={len(trainset)}")
    print(f"[Run config] seed_prompt={seed_candidate['system_prompt']}")
    print(
        "[Run config] retries="
        f"{args.max_retries} backoff={args.initial_backoff_seconds}s..{args.max_backoff_seconds}s "
        f"timeout={args.timeout_seconds}s min_call_interval={args.min_call_interval_seconds}s"
    )
    task_lm_callable = make_retry_lm_callable(
        model_name=MODEL_NAME,
        base_url=BASE_URL,
        api_key=API_KEY,
        max_retries=args.max_retries,
        initial_backoff_seconds=args.initial_backoff_seconds,
        max_backoff_seconds=args.max_backoff_seconds,
        timeout_seconds=args.timeout_seconds,
        min_call_interval_seconds=args.min_call_interval_seconds,
    )
    reflection_lm_callable = make_retry_lm_callable(
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
        valset=trainset,
        task_lm=task_lm_callable,
        reflection_lm=reflection_lm_callable,
        evaluator=ExactMatchEvaluator(),
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        run_dir=str(run_dir),
        display_progress_bar=True,
    )

    seed_score = result.val_aggregate_scores[0]
    best_score = result.val_aggregate_scores[result.best_idx]
    improvement = best_score - seed_score

    summary = {
        "timestamp": timestamp,
        "tag": args.tag,
        "run_dir": str(run_dir),
        "base_url": BASE_URL,
        "task_lm": MODEL_NAME,
        "reflection_lm": MODEL_NAME,
        "max_metric_calls": args.max_metric_calls,
        "reflection_minibatch_size": args.reflection_minibatch_size,
        "max_retries": args.max_retries,
        "initial_backoff_seconds": args.initial_backoff_seconds,
        "max_backoff_seconds": args.max_backoff_seconds,
        "timeout_seconds": args.timeout_seconds,
        "min_call_interval_seconds": args.min_call_interval_seconds,
        "dataset_profile": args.dataset_profile,
        "dataset_size": len(trainset),
        "seed_candidate": seed_candidate,
        "seed_score": seed_score,
        "best_score": best_score,
        "improvement": improvement,
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "best_idx": result.best_idx,
        "best_candidate": result.best_candidate,
        "val_aggregate_scores": result.val_aggregate_scores,
    }

    summary_json_path = run_dir / "summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary_md_path = run_dir / "summary.md"
    summary_md = "\n".join(
        [
            "# GEPA One-Round Summary",
            f"- Run Dir: `{run_dir}`",
            f"- Base URL: `{BASE_URL}`",
            f"- Task LM: `{MODEL_NAME}`",
            f"- Reflection LM: `{MODEL_NAME}`",
            f"- Max Metric Calls: `{args.max_metric_calls}`",
            f"- Reflection Minibatch Size: `{args.reflection_minibatch_size}`",
            f"- Seed Score: `{seed_score:.4f}`",
            f"- Best Score: `{best_score:.4f}`",
            f"- Improvement: `{improvement:.4f}`",
            f"- Num Candidates: `{result.num_candidates}`",
            f"- Total Metric Calls: `{result.total_metric_calls}`",
            "",
            "## Best Candidate",
            "```json",
            json.dumps(result.best_candidate, ensure_ascii=False, indent=2),
            "```",
        ]
    )
    summary_md_path.write_text(summary_md, encoding="utf-8")

    print("=== One-Round GEPA Run Complete ===")
    print(f"Run dir: {run_dir}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary Markdown: {summary_md_path}")
    print(f"Seed score: {seed_score:.4f}")
    print(f"Best score: {best_score:.4f}")
    print(f"Improvement: {improvement:.4f}")


if __name__ == "__main__":
    main()
