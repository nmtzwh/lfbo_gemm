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
  --space-config examples/space_config.yaml \
  --objective one_shot --budget 64 --history run.jsonl --output plans.json
```

LFBO begins with random plans, labels the best observed quantile as good,
trains a Random Forest classifier, and filters multi-field neighbors around the
best search copies. Rejections and process failures are retained as negative
training examples. See `docs/external_tuner_design.md` in the parent workspace
for the full workflow and algorithm rationale, and `docs/search_space.md` for
the blocking, scheduling, packing, and microkernel parameter space.

## Custom search space

Pass `--space-config FILE` to replace selected built-in domains with a
versioned YAML or JSON `SpaceConfig`. Unspecified domains keep their built-in
values. Explicit domains are strictly bounded by default, while the runtime
oneDNN baseline is still measured for comparison. Set `inherit_baseline: true`
to add its value to each explicitly configured domain and keep an in-domain
baseline eligible for selection.

The configuration supports conditional forcing and policy-side limits for
scratchpad and minimum parallel work. Runner capabilities remain authoritative:
unsupported categorical or microkernel values fail before tuning, and every
candidate is still finalized and validated by the native runner. See
`examples/space_config.yaml` and the parent repository's
`docs/search_space.md`.

The effective domains, canonical configuration, and `space_hash` are written
to the result. An explicit configuration is included in the JSONL history
fingerprint, so changing it rejects resume against an older history.

## Console progress

The default command keeps stdout machine-readable and prints only the final
JSON object. Add `-v` for compact human-readable progress on stderr:

```bash
uv run matopt-tuner tune ... -v
```

This reports the workload and ISA, effective search domains, baseline, one
summary per LFBO generation, current best parameters and improvement, and the
final selected plan. Use `-vv` to include every benchmarked or rejected
candidate. Important values are colored when stderr is a terminal. Use
`--color always` to preserve color through a log capture or `--color never` to
disable ANSI sequences. Setting the standard `NO_COLOR` environment variable
also disables automatic color.

Verbose diagnostics never go to stdout, so scripts may continue parsing the
final JSON response unchanged.

## AOT commands

`matopt-tuner export` consumes a selected tuning result and enforces the
`aot_bundle_v1` native capture contract. It validates every image hash and
ordinal, generates the fixed-shape public header and CMake package, asks the
runner for fresh-process zero-JIT validation, and publishes by atomic rename.
The package embeds the images in ELF executable text and reuses the ordinary
oneDNN MatMul driver. It rejects older runners instead of falling back to
runtime JIT.

`matopt-tuner benchmark` configures the standalone project in `../benchmark`,
requires system OpenBLAS, and emits `correctness.json`,
`google-benchmark.json`, and `summary.json`. Google Benchmark v1.9.5 is fetched
only when `--fetch-google-benchmark` is explicit.

## Optimization trajectory

Render benchmarked latencies in evaluation order, colored by LFBO generation,
with the generation-level best-so-far Pareto envelope overlaid:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra lfbo --extra visualization

UV_CACHE_DIR=/tmp/uv-cache uv run matopt-tuner visualize \
  --history run.jsonl --output trajectory.png --metric one_shot
```

`--metric` accepts `one_shot`, `steady`, or `median`. The visualizer reads the
append-only history directly and tolerates an incomplete final JSONL record.
