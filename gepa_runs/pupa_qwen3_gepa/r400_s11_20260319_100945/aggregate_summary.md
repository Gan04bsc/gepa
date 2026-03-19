# PUPA GEPA Experiment Summary

## Config
- Task: `PUPA`
- Model: `openai/qwen3-8b`
- Optimizer: `GEPA` (Merge disabled)
- Budget: `400` rollouts
- Reflection minibatch: `3`
- Split: `111 / 60 / 221`
- Seeds: `[11]`

## Requested Metrics
- Best val score (mean +/- std): `0.4054 +/- 0.0000`
- Final test score (mean +/- std): `0.3948 +/- 0.0000`
- Final prompt length chars (mean +/- std): `3481.0 +/- 0.0`

## Module Updates
- craft_redacted_request updates (mean +/- std): `3.00 +/- 0.00`
- respond_to_query updates (mean +/- std): `2.00 +/- 0.00`

## Feedback Categories (accepted updates)
- privacy_or_pii: `5`
- response_quality: `2`

## Artifacts
- Aggregate JSON: `D:\study\Agent\浙大APO\gepa\gepa_runs\pupa_qwen3_gepa\r400_s11_20260319_100945\aggregate_summary.json`
- Per-seed summary: `D:\study\Agent\浙大APO\gepa\gepa_runs\pupa_qwen3_gepa\r400_s11_20260319_100945/seed_*/seed_summary.json`
- Accepted updates: `D:\study\Agent\浙大APO\gepa\gepa_runs\pupa_qwen3_gepa\r400_s11_20260319_100945/seed_*/accepted_updates.json`