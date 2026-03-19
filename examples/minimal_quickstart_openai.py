"""Minimal GEPA example using real OpenAI APIs via LiteLLM.

Run:
    uv run python -m examples.minimal_quickstart_openai
"""

from __future__ import annotations

import os

import gepa
from gepa.adapters.default_adapter.default_adapter import DefaultDataInst, EvaluationResult

# Fill your key here before running.
API_KEY = "sk-FPKCGTdIXqF3pTTN7eBcFa5535184528Bb876d52799608E4"
BASE_URL = "https://api.shubiaobiao.cn/v1"
MODEL_NAME = "openai/gpt-4.1-mini"


class ExactMatchEvaluator:
    """Simple evaluator for short factual answers."""

    def __call__(self, data: DefaultDataInst, response: str) -> EvaluationResult:
        gold = data["answer"].strip().lower()
        pred = response.strip().lower()

        # Accept exact match or common "final answer: <x>" style.
        pred_last_token = pred.split()[-1] if pred else ""
        is_correct = pred == gold or pred_last_token == gold

        score = 1.0 if is_correct else 0.0
        if is_correct:
            feedback = f"Correct. Expected '{data['answer']}', got '{response}'."
        else:
            feedback = (
                f"Incorrect. Expected exact answer '{data['answer']}', got '{response}'. "
                "Return the final answer as a single short token."
            )
        return EvaluationResult(score=score, feedback=feedback, objective_scores=None)


def main() -> None:
    if API_KEY == "YOUR_API_KEY_HERE":
        raise SystemExit("Please set API_KEY in this script before running.")

    # LiteLLM reads OpenAI-compatible settings from environment variables.
    os.environ["OPENAI_API_KEY"] = API_KEY
    os.environ["OPENAI_API_BASE"] = BASE_URL

    trainset: list[DefaultDataInst] = [
        {"input": "What is 7 + 5? Reply with only the number.", "answer": "12", "additional_context": {}},
        {"input": "What is 9 - 4? Reply with only the number.", "answer": "5", "additional_context": {}},
        {"input": "What is 3 * 6? Reply with only the number.", "answer": "18", "additional_context": {}},
        {"input": "What is the capital of France? Reply with one word.", "answer": "Paris", "additional_context": {}},
    ]

    seed_candidate = {
        "system_prompt": "You are a helpful assistant. Answer the user query.",
    }

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=trainset,
        task_lm=MODEL_NAME,
        reflection_lm=MODEL_NAME,
        evaluator=ExactMatchEvaluator(),
        max_metric_calls=30,
        reflection_minibatch_size=2,
        run_dir="gepa_runs/openai_minimal",
        display_progress_bar=True,
    )

    seed_score = result.val_aggregate_scores[0]
    best_score = result.val_aggregate_scores[result.best_idx]

    print("=== GEPA OpenAI Minimal Demo ===")
    print(f"Seed score: {seed_score:.3f}")
    print(f"Best score: {best_score:.3f}")
    print(f"Improvement: {best_score - seed_score:.3f}")
    print(f"Candidates explored: {result.num_candidates}")
    print(f"Total metric calls: {result.total_metric_calls}")
    print("Best candidate:")
    print(result.best_candidate)


if __name__ == "__main__":
    main()
