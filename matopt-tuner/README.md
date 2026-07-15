# MatOpt Tuner

This package owns search and persistence for the patched oneDNN
`matopt-runner`. The native process remains authoritative for affinity, plan
validation, correctness, and timing.

```bash
PYTHONPATH=src ONEDNN_MAX_CPU_ISA=AVX2 python3 -m matopt.cli tune \
  --runner ../oneDNN/build/tools/matopt/matopt-runner \
  --m 256 --n 256 --k 256 --threads 1 --cpus 0 \
  --objective one_shot --budget 16 --history run.jsonl --output plans.json
```

Each evaluation uses a fresh runner process. A signal or timeout is recorded
against the exact proposed plan before tuning continues.

## LFBO Pattern Search

Install the optional scikit-learn environment and run the Helion-inspired LFBO
strategy with:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra lfbo

UV_CACHE_DIR=/tmp/uv-cache ONEDNN_MAX_CPU_ISA=AVX2 uv run matopt-tuner tune \
  --search lfbo --runner /path/to/matopt-runner \
  --m 256 --n 256 --k 256 --threads 4 --cpus 0-3 \
  --objective one_shot --budget 64 --history run.jsonl --output plans.json
```

LFBO begins with random plans, labels the best observed quantile as good,
trains a Random Forest classifier, and filters multi-field neighbors around the
best search copies. Rejections and process failures are retained as negative
training examples. See `docs/external_tuner_design.md` in the parent workspace
for the full workflow and algorithm rationale.
