# PUPA GEPA Experiment

This folder provides a runnable PUPA-style GEPA pipeline with:
- 2 optimized modules: `craft_redacted_request`, `respond_to_query`
- multi-seed run
- automatic aggregation of all requested metrics

## 1) Dataset Format

Supported input formats: `.json` or `.jsonl`.

Each record should contain:
- query field: one of `user_query` / `query` / `input` / `private_query` / `original_query`
- answer field (optional but recommended): one of `reference_answer` / `reference_response` / `answer` / `target_answer` / `target_response` / `response`
- pii field (optional): one of `pii_tokens` / `pii` / `sensitive_terms` / `private_entities` / `sensitive_spans`

## 2) Run (your requested config)

```powershell
set UV_CACHE_DIR=.uv_cache
uv run python -m examples.pupa_experiment.run_pupa_qwen3_gepa `
  --model openai/qwen3-8b `
  --train-file <path_to_train.jsonl> `
  --val-file <path_to_val.jsonl> `
  --test-file <path_to_test.jsonl> `
  --train-size 111 `
  --val-size 60 `
  --test-size 221 `
  --max-metric-calls 1200 `
  --reflection-minibatch-size 3 `
  --seeds 11,22,33 `
  --max-retries 30 `
  --initial-backoff-seconds 2.5 `
  --min-call-interval-seconds 1.2 `
  --tag pupa_qwen3_align
```

`--seeds` can include more than 3 seeds.
If you have strict rate limits, increase `--min-call-interval-seconds` (for example, `1.5` to `2.0`).

## 3) Resume From Interrupted Run

GEPA saves state files in each `seed_*` directory. To resume, pass the same root directory:

```powershell
uv run python -m examples.pupa_experiment.run_pupa_qwen3_gepa `
  --resume-run-dir gepa_runs/pupa_qwen3_gepa/<timestamp>_<tag> `
  --model openai/qwen3-8b `
  --thinking-mode off `
  --train-file data/pupa/train.jsonl `
  --val-file data/pupa/val.jsonl `
  --test-file data/pupa/test.jsonl `
  --train-size 111 `
  --val-size 60 `
  --test-size 221 `
  --max-metric-calls 1200 `
  --reflection-minibatch-size 3 `
  --seeds 11,22,33
```

## 4) Dry-run (no API)

```powershell
uv run python -m examples.pupa_experiment.run_pupa_qwen3_gepa `
  --dry-run `
  --train-size 12 `
  --val-size 6 `
  --test-size 8 `
  --max-metric-calls 40 `
  --seeds 1,2,3 `
  --tag dryrun
```

## 5) Output

All artifacts are under:
- `gepa_runs/pupa_qwen3_gepa/<timestamp>_<tag>/`

Key files:
- `aggregate_summary.json`
- `aggregate_summary.md`
- `seed_*/seed_summary.json`
- `seed_*/accepted_updates.json`

`seed_*/accepted_updates.json` records each accepted update with:
- `iteration`
- `module`
- `feedback_categories`
- `num_reflective_records`
