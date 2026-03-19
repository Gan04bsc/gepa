"""Prepare train/val/test JSONL splits from a single PUPA CSV file.

Expected CSV columns:
- user_query
- target_response
- pii_units (optional, separated by "||")
- conversation_hash (optional)
- predicted_category (optional)
- redacted_query (optional)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PUPA CSV into train/val/test JSONL files.")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to source CSV.")
    parser.add_argument("--output-dir", type=str, default="data/pupa", help="Directory for output JSONL files.")
    parser.add_argument("--train-size", type=int, default=111)
    parser.add_argument("--val-size", type=int, default=60)
    parser.add_argument("--test-size", type=int, default=221)
    parser.add_argument("--split-seed", type=int, default=0, help="Seed for deterministic split.")
    return parser.parse_args()


def split_pii_units(text: str) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in text.split("||") if x.strip()]


def normalize_row(row: dict[str, str]) -> dict[str, object]:
    user_query = (row.get("user_query") or "").strip()
    target_response = (row.get("target_response") or "").strip()
    if not user_query:
        raise ValueError("Missing required field: user_query")

    pii_tokens = split_pii_units((row.get("pii_units") or "").strip())

    return {
        "user_query": user_query,
        "reference_answer": target_response,
        "pii_tokens": pii_tokens,
        "additional_context": {
            "conversation_hash": (row.get("conversation_hash") or "").strip(),
            "predicted_category": (row.get("predicted_category") or "").strip(),
            "redacted_query": (row.get("redacted_query") or "").strip(),
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    data = [normalize_row(r) for r in raw_rows]
    total_needed = args.train_size + args.val_size + args.test_size
    if len(data) < total_needed:
        raise ValueError(f"Not enough rows: have {len(data)}, need {total_needed}.")

    rng = random.Random(args.split_seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    train_idx = indices[: args.train_size]
    val_idx = indices[args.train_size : args.train_size + args.val_size]
    test_idx = indices[args.train_size + args.val_size : total_needed]
    unused_idx = indices[total_needed:]

    train_rows = [data[i] for i in train_idx]
    val_rows = [data[i] for i in val_idx]
    test_rows = [data[i] for i in test_idx]

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"
    manifest_path = output_dir / "split_manifest.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(test_path, test_rows)

    manifest = {
        "input_csv": str(input_csv),
        "split_seed": args.split_seed,
        "source_rows": len(data),
        "requested_sizes": {
            "train": args.train_size,
            "val": args.val_size,
            "test": args.test_size,
        },
        "written_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "unused": len(unused_idx),
        },
        "output_files": {
            "train_file": str(train_path),
            "val_file": str(val_path),
            "test_file": str(test_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Done] Wrote dataset splits:")
    print(f"- {train_path} ({len(train_rows)})")
    print(f"- {val_path} ({len(val_rows)})")
    print(f"- {test_path} ({len(test_rows)})")
    print(f"- {manifest_path}")


if __name__ == "__main__":
    main()
