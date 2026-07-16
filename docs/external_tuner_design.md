# External MatOpt Tuner Architecture

## Decision

Refactor MatOpt into two deliverables:

1. a small, version-pinned oneDNN patch that exposes the existing BRGEMM
   MatMul configuration seam through a non-installed native runner; and
2. an out-of-tree Python package that owns search, history, Pareto selection,
   and user-facing tuning workflows.

The boundary is a versioned JSON process protocol. Python does not load an
internal oneDNN shared library and the patch does not depend on Python. This
keeps oneDNN's public ABI unchanged, prevents Python/GIL state from affecting
measurements, lets the runner establish affinity before OpenMP initialization,
and contains a crashing candidate to a child process.

```text
Python tuner                         patched oneDNN build

search space / LFBO  -- request --> matopt-runner
history / resume                    affinity and identity
feasibility model                   plan validation/finalization
Pareto / selection     <-- result -- existing BRGEMM MatMul primitive
reporting                            correctness and measurement
```

The current native exhaustive search remains useful as a reference during the
migration, but it is not part of the long-term oneDNN patch.

## Implementation status

The first migration boundary is implemented:

- the patched build produces both the compatibility `matopt` executable and
  the process-oriented `matopt-runner`;
- `matopt-runner` implements `capabilities`, `baseline`, `evaluate`, and
  request-based `inspect` operations;
- the independent `matopt-tuner/` Python package implements subprocess crash
  isolation, capability-derived macro and micro candidates, append-only JSONL
  resume, Pareto/noise-aware selection, a reference random strategy, and a
  generic LFBO `ask`/`tell` adapter; and
- native `tune` remains temporarily for regression comparison.

The standalone package now includes an LFBO Pattern Search implementation using
NumPy and scikit-learn behind the `lfbo` optional extra. The generic external
`ask`/`tell` adapter remains available for other optimizers.

## Ownership boundary

### oneDNN patch

The patch contains only mechanisms that require access to oneDNN internals:

- schema-versioned `plan_t` and `finalized_plan_t`;
- the scoped, thread-local plan adapter;
- post-heuristic macro override and deterministic re-finalization;
- full-descriptor microkernel hints;
- capture of the default and realized configurations;
- architecture-specific validation, including the temporary AArch64
  `nthr_k=1` restriction;
- a dense FP32 AArch64 padded-A row-copy fallback while the upstream SVE
  copy-A JIT remains unfinished;
- construction and execution of the ordinary oneDNN primitive;
- persistent-weight reorder, correctness validation, timing, and scratchpad
  accounting;
- CPU affinity, worker-affinity verification, machine/build identity, and
  primitive-cache disablement; and
- a non-installed `matopt-runner` executable.

It must not contain candidate generation, search ranking, LFBO, Pareto
selection, history policy, or Python bindings. The normal null-plan production
path remains unchanged.

### Python package

The out-of-tree package owns policy:

- search-space construction from runner-reported capabilities;
- baseline mutations, random search, grid search, and LFBO `ask`/`tell` loops;
- categorical and integer feature encoding;
- pruning that does not require backend-private derived state;
- deterministic proposal order and seeds;
- append-only history, resume, retry, and crash classification;
- multi-fidelity scheduling of macro, micro, and finalist measurements;
- instability handling, Pareto construction, baseline-noise fallback, and
  selected-plan output; and
- reports and plots.

Python may predict feasibility, but the native runner is authoritative. A
plan is never considered valid until the runner accepts and finalizes it.

## Native runner

Use a narrow executable interface rather than `pybind11`, `ctypes`, or a new
installed oneDNN API. Its stdout is exactly one JSON response; diagnostics go
to stderr. The initial commands are:

```text
matopt-runner capabilities --request workload.json
matopt-runner baseline     --request workload.json
matopt-runner evaluate     --request evaluation.json
matopt-runner inspect      --request plan.json
```

`capabilities` performs no timing. It returns the protocol version, plan
schema, supported workload domain, legal packing policies, backend constraints,
micro-hint domains, scratchpad limit, and machine/build fingerprint.

`baseline` captures the current heuristic, constructs the safe effective
baseline, replays it, validates it numerically, and measures it with the
requested measurement profile. On AArch64 the effective baseline fixes
`nthr_k=1` while split-K is disabled. Both the captured native default and the
effective baseline are returned so the exception is explicit.

`evaluate` accepts exactly one plan. It validates, finalizes, checks numerical
correctness, and measures that plan. It never searches or silently changes a
field. Primitive/JIT creation remains outside timed regions.

`inspect` validates and finalizes without executing timed samples. It is used
for debugging and cheap constraint discovery, not as a substitute for
`evaluate` acceptance.

Process-per-evaluation is the default. The Python parent appends a `started`
record before spawning the runner and classifies signals, timeout, malformed
output, and nonzero exits. This directly addresses backend/JIT faults without
losing the proposed point. A future `serve` mode may reduce launch overhead,
but is opt-in and must provide equivalent candidate isolation before it can be
used for unattended tuning.

## Protocol

Use canonical UTF-8 JSON with integer schema versions. JSON is appropriate for
the low request rate and preserves inspectability in support bundles. Do not
serialize native structs or depend on enum ordinals.

An evaluation request has four sections:

```json
{
  "protocol_version": 1,
  "request_id": "stable-uuid",
  "expected_fingerprint": "sha256:...",
  "workload": {
    "m": 256,
    "n": 256,
    "k": 256,
    "dtype": "f32",
    "layout": "dense_row_major",
    "alpha": 1.0,
    "beta": 0.0,
    "threads": 4,
    "cpus": "144-147"
  },
  "plan": {
    "schema_version": 1,
    "m_blk": 128,
    "n_blk": 64,
    "k_blk": 256,
    "m_chunk_size": 1,
    "n_chunk_size": 2,
    "k_chunk_size": 1,
    "brgemm_batch_size": 1,
    "nthr_k": 1,
    "pack_a": "direct",
    "pack_b": "direct",
    "bd_block": 4,
    "ld_block2": 2,
    "loop_order": "default"
  },
  "measurement": {
    "profile": "macro",
    "warmups": 3,
    "samples": 3,
    "minimum_sample_ms": 100,
    "seed": 19260817
  }
}
```

The runner returns one terminal status:

- `benchmarked`: finalized plan, correctness, measurements, and fingerprint;
- `rejected`: stable reason code plus a human-readable detail;
- `incorrect`: mismatch summary and finalized plan; or
- `error`: runner/infrastructure failure that is not a plan rejection.

The Python parent adds `crashed`, `timed_out`, and `protocol_error`, because a
terminated runner cannot produce those responses. Reason codes, unlike detail
strings, are API: for example `split_k_unsupported`, `invalid_k_granularity`,
`packing_layout_mismatch`, `insufficient_k_chunks`, and
`scratchpad_limit_exceeded`.

Every response repeats `request_id`, protocol version, plan schema, and the
complete fingerprint. The runner rejects a nonempty `expected_fingerprint`
that differs from its own. Unknown required fields or enum values are errors;
unknown optional fields are accepted only within a declared protocol minor
version.

## Fingerprint and persistence

The runner creates the authoritative fingerprint from:

- workload and exact CPU mask;
- CPU model, effective ISA and vector width;
- cache topology and sizes;
- thread count and OpenMP runtime identity;
- oneDNN source hash plus patch-set identifier;
- compiler, flags relevant to code generation, and build type;
- runner protocol version and plan schema; and
- capability constraints such as AArch64 split-K support.

The Python history is append-only JSONL. Each record contains the canonical
request, proposal metadata, runner response or parent-classified failure,
timestamps, and search-state checkpoint reference. A truncated final line is
recoverable; any earlier malformed line or fingerprint mismatch rejects
resume. The final compact JSON contains the workload, fingerprint, baseline,
selected plan per objective, Pareto set, and enough measurements to audit the
choice.

Search-library state is a cache, not the source of truth. On resume, the tuner
reconstructs observations from JSONL and then restores or refits the LFBO
model. This avoids coupling durable results to a particular Python or model
serialization version.

## Python API and package layout

The public Python surface should be library-first, with a thin CLI:

```python
runner = MatOptRunner("/path/to/matopt-runner")
session = TuningSession(workload, runner, history="run.jsonl")
result = session.tune(search=LFBOConfig(budget=256), objectives=["one_shot"])
```

Suggested repository layout:

```text
matopt-tuner/
  pyproject.toml
  src/matopt/
    protocol.py       # typed requests/responses and canonical JSON
    runner.py         # subprocess, timeout, signals, stdout validation
    history.py        # JSONL transaction and resume checks
    space.py          # capability-derived conditional search space
    objectives.py     # metrics, noise threshold, Pareto selection
    schedulers.py     # macro/micro/finalist fidelity policy
    search/
      random.py
      grid.py
      lfbo.py
    cli.py
  tests/
```

The core package should depend only on a lightweight schema/typing stack.
Machine-learning dependencies belong in an optional extra such as
`matopt-tuner[lfbo]`, so baseline and random tuning remain easy to deploy.

## LFBO Pattern Search

The initial implementation follows the workflow described in PyTorch's
[Helion LFBO autotuning article](https://pytorch.org/blog/accelerating-autotuning-in-helion/)
and its published `LFBOPatternSearch` implementation. It uses classification
rather than latency regression: the model only needs to rank promising plans,
can learn directly from invalid plans, and does not spend capacity fitting the
latency surface of clearly poor configurations.

### Algorithm selection

The surrogate is `sklearn.ensemble.RandomForestClassifier` with log-loss, 100
trees, and deterministic per-generation seeds. It uses one fitting worker by
default: unlike Helion's GPU use case, consuming every CPU core immediately
before a CPU MatMul measurement can heat or contend with the pinned target
cores. `--lfbo-model-jobs` may be increased on a host with isolated tuning and
model-fitting CPU sets. Random Forest was selected because:

- the plan space is small-data, mixed discrete, non-smooth, and conditional;
- trees accept ordered and one-hot features without differentiable kernels;
- training cost is low compared with native candidate measurements;
- class probabilities provide the LFBO ranking score; and
- leaf assignments provide a useful model-aware similarity metric at no extra
  model-training cost.

Gaussian-process BO is deferred because categorical packing/layout choices,
hard invalid regions, and more than ten interacting fields require a more
complex kernel and acquisition optimizer. Direct latency regression is not the
default because rejected, incorrect, crashed, and timed-out candidates have no
meaningful latency target. Gradient boosting and neural classifiers remain
possible later, but Random Forest gives the diversity signal used by the Helion
workflow.

### Features and canonical plans

`PlanSpace` derives every domain from the native capability response and the
captured baseline. Ordered fields are represented by normalized domain index:

```text
M_blk, N_blk, K_blk, N_chunk_size, BRGEMM batch size, nthr_k,
bd_block, ld_block2
```

Packing policies and loop order use one-hot encoding. The plan is canonicalized
before hashing or encoding: n32/n64 B policies force the matching `N_blk`, and
unsupported AArch64 split-K values never enter the domain. The native runner
still performs authoritative validation; Python canonicalization is only a
cheap search-space reduction.

### Search workflow

For one workload, CPU mask, objective, and runner fingerprint:

1. Capture, replay, validate, and benchmark the oneDNN baseline. Seed it into
   the observation set without charging it to the LFBO proposal budget.
2. Benchmark up to 24 deterministic random initial plans. Previous terminal
   JSONL records are restored first and count against the requested budget.
3. Convert each valid observation to a minimization loss: one-shot or steady
   milliseconds directly, and reciprocal GFLOP/s for throughput.
4. Compute the 10th percentile of finite losses. Plans at or below it are
   positive (`good`); all other plans are negative. Rejected, incorrect,
   unstable-infrastructure, crashed, and timed-out plans receive infinite loss
   and therefore negative labels.
5. Weight positive samples by improvement below the quantile threshold,
   normalized to mean weight one. Negative samples have weight one. If only one
   class exists, skip fitting and select randomly until both classes appear.
6. Keep the three best finite observations as search copies. Generate up to 256
   candidates around them with random perturbations across one or two fields.
   Ordered fields move by at most two domain indices; categorical fields may
   switch category. This permits larger jumps than exhaustive one-field pattern
   search.
7. Fit the classifier on every observation collected so far. Score candidates
   by probability of the positive class.
8. Select 10% of the candidate pool sequentially. The first plan has maximum
   positive probability. Each later plan maximizes:

   ```text
   P(good | plan) - similarity_penalty * mean_leaf_similarity
   ```

   Leaf similarity is the fraction of Random Forest trees in which two plans
   land in the same leaf. The default penalty is 1.0.
9. Benchmark the selected batch serially on the pinned CPU mask, append every
   terminal result, refit from all observations, update search copies, and
   repeat for at most eight generations.
10. Stop after two generations below 0.1% best-loss improvement, exhaustion of
    novel neighbors, the generation limit, or the total evaluation budget.
    Pareto construction, finalist measurement, and baseline-noise fallback
    remain independent of the surrogate.

The parameters are exposed through `--lfbo-*` CLI options. Defaults are starting
points rather than universal constants; target-host studies must compare them
against deterministic random search at the same native evaluation budget.

### Persistence and failures

The model is never persisted as authoritative state. Resume reads terminal
JSONL records in timestamp order, restores the used budget and visited hashes,
and refits scikit-learn state. This prevents pickle/library-version coupling.
The exact runner response remains attached to negative observations, so a
future model may distinguish static plan rejection from a process crash.

Unlike the earlier placeholder design, the first classifier combines quality
and feasibility: every non-top-quantile point is negative, including failures.
This matches Helion's classification rationale and avoids an undertrained
second model. A separate feasibility classifier is a future option if invalid
plans dominate enough to hide quality differences among valid plans.

Install and run with:

```text
uv sync --extra lfbo
uv run matopt-tuner tune --search lfbo ...
```

### Trajectory visualization

The standalone tuner renders an existing JSONL history without loading oneDNN
or changing the history. `matopt-tuner visualize` orders correct benchmarked
records by timestamp and plots:

- one scatter point per measured plan on the evaluation timeline;
- scatter color as the recorded LFBO generation;
- the captured oneDNN baseline as a distinct star; and
- a stepwise Pareto envelope containing the cumulative minimum latency at the
  end of each generation.

The plotted metric can be one-shot, steady-state, or median latency. Rejected,
incorrect, and failed candidates have no meaningful latency and are omitted
from the scatter, while remaining available to LFBO through the unchanged
JSONL history. A truncated final JSONL record is ignored for visualization;
complete malformed records and mixed machine fingerprints are rejected.

```text
uv sync --extra lfbo --extra visualization
uv run matopt-tuner visualize \
  --history run.jsonl --output trajectory.png --metric one_shot
```

## Measurement and concurrency rules

Only one evaluation may run on a CPU mask at a time. Parallel proposal
generation is allowed, but concurrent measurements on overlapping masks are
not. The runner sets process affinity before creating an engine, applies
close/core OpenMP placement, verifies observed workers, disables the oneDNN
primitive cache, and reports violations as infrastructure errors.

The runner, not Python, defines timing semantics. Protocol measurement profiles
expand to explicit warmup, sample, and duration fields in history so defaults
can evolve without changing old results. Persistent B reorder, allocation,
JIT exclusion, validation inputs, stability, and GFLOP/s follow the audited MVP
rules in `design.md`.

## Migration plan

### Phase 1: extract the evaluator

- Rename the executable to `matopt-runner` and retain `benchmark`/`inspect`.
- Split native files into backend adapter, protocol/identity, evaluator, and a
  small main program.
- Add `capabilities`, `baseline`, and single-plan `evaluate` commands.
- Keep the existing native `tune` command temporarily as a regression oracle.
- Add golden protocol fixtures and exit/signal behavior tests.

Stop when the external Python grid search produces the same candidate hashes,
measurements, Pareto set, and selections as native `tune` for deterministic
test fixtures.

### Phase 2: move policy to Python

- Port candidate generation, pruning, history, Pareto selection, and baseline
  fallback without adding ML dependencies.
- Test interrupted resume, malformed output, timeout, simulated signal death,
  fingerprint mismatch, and AArch64 `nthr_k>1` rejection.
- Remove native search code only after cross-language equivalence tests pass.

### Phase 3: add LFBO

- Implement the common `ask`/`tell` search interface and a random-search
  reference implementation first.
- Add LFBO behind an optional dependency extra.
- Compare LFBO against deterministic random search at equal evaluation budgets;
  correctness and resume behavior remain gates, performance does not.

### Phase 4: package for end users

- Export a oneDNN patch series pinned to a supported upstream tag and hash.
- Publish the independent Python source distribution and lock/test its optional
  LFBO environment.
- Provide a build helper that applies the patch, verifies the source hash,
  configures static oneDNN with OpenMP and `DNNL_BUILD_MATOPT=ON`, and prints
  the resulting fingerprint.
- Include a compatibility matrix of oneDNN tag, patch-set ID, runner protocol,
  plan schema, target ISA, and tested compiler.

Do not promise that a patch applies to arbitrary oneDNN revisions. A changed
post-heuristic seam requires a newly audited patch and a new patch-set ID.

## Acceptance

The refactor is complete when:

- a normal oneDNN build and null-plan MatMul path are unchanged;
- the patch builds without Python and the Python package installs without
  oneDNN headers;
- every explicit plan is either realized exactly or rejected with a stable
  reason code;
- runner crashes and timeouts preserve the exact proposed point in history;
- native and Python reference searches agree on canonical plan hashes,
  measurements within noise, Pareto membership, and baseline fallback;
- resume rejects environment drift and reconstructs search observations from
  JSONL;
- the same Python tuner works with SVE256 and AVX-512 runners using reported
  capabilities; and
- LFBO can be replaced by another `ask`/`tell` strategy without changing the
  oneDNN patch or runner protocol.
