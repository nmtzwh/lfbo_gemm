# MatOpt: offline BRGEMM MatMul tuning for oneDNN

This repository contains two distributable components:

- `matopt-tuner/`: the standalone Python search, history, and persistence layer;
- `patches/onednn-v3.12.2-matopt.patch`: the native MatOpt backend and runner
  patch for oneDNN v3.12.2 (`76ef43d856388d95c1e23573a7d057766e5ee00e`).

The native runner owns affinity, oneDNN descriptor construction, plan
validation, correctness checks, and timing. The Python package owns candidate
generation, LFBO or random search, append-only history, resume, and Pareto
selection.

## Apply and build the oneDNN patch

```bash
git clone --branch v3.12.2 https://github.com/uxlfoundation/oneDNN.git
cd oneDNN
git apply --check ../patches/onednn-v3.12.2-matopt.patch
git apply ../patches/onednn-v3.12.2-matopt.patch

cmake -S . -B build \
  -DDNNL_BUILD_MATOPT=ON \
  -DDNNL_LIBRARY_TYPE=STATIC \
  -DDNNL_CPU_RUNTIME=OMP \
  -DONEDNN_BUILD_GRAPH=OFF \
  '-DDNNL_ENABLE_PRIMITIVE=MATMUL;REORDER'
cmake --build build --target matopt-runner
```

`MATMUL;REORDER` is required because persistent B policies use a one-time
oneDNN reorder into blocked weights. The patch is internal and does not change
oneDNN's public ABI.

## Install and run the Python tuner

```bash
cd matopt-tuner
uv sync --extra lfbo

uv run matopt-tuner tune \
  --search lfbo \
  --runner ../oneDNN/build/tools/matopt/matopt-runner \
  --m 256 --n 256 --k 256 \
  --threads 4 --cpus 0-3 \
  --objective one_shot --budget 64 \
  --history run.jsonl --output plans.json
```

An explicit Linux CPU mask is required. See
`docs/external_tuner_design.md` for the architecture and LFBO workflow, and
`docs/design.md` for the audited native-backend design.

## Test

```bash
cd matopt-tuner
uv run python -m unittest discover -s tests -v
```

Native integration tests can use an already built runner:

```bash
MATOPT_RUNNER=/path/to/matopt-runner MATOPT_CPU=0 \
  uv run python -m unittest discover -s tests -v
```

Plot an LFBO optimization trajectory from its append-only history with:

```bash
cd matopt-tuner
uv sync --extra lfbo --extra visualization
uv run matopt-tuner visualize \
  --history run.jsonl --output trajectory.png --metric one_shot
```
