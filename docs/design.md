# Audited BRGEMM MatMul Offline Tuner MVP

This design is tied to oneDNN v3.12.2, commit `76ef43d`. `matopt` is an
offline, opt-in tuning executable for dense FP32 MatMul on AArch64 SVE256 and
x64 AVX-512. The local AVX2 machine is a compilation and platform-independent
test host, not a performance-validation host.

## Scope and invariants

The MVP supports batch 1, dense row-major A/B/C, `alpha=1`, `beta=0`, a fixed
intra-op thread count, and an explicit Linux CPU mask. It tunes the existing
architecture-specific BRGEMM MatMul primitive. It does not copy the driver,
tail handling, copy kernels, JIT kernel ownership, or scratchpad management.

`beta=1` is not treated as a kernel switch: oneDNN MatMul requires a sum
post-op to preserve the old destination. Post-ops, transposes, non-dense
strides, batch dimensions, quantized types, BF16/AMX, PMU collection, runtime
plan lookup, and plan-family reuse are outside this MVP.

The following controls are also deferred because they need separate kernel
and validation work: `rd_block`, software-prefetch distance, non-temporal
loads/stores, store interleaving, and AOT machine code.

## oneDNN audit findings

The integration points are:

- `src/cpu/x64/matmul/brgemm_matmul_utils.cpp` and
  `src/cpu/aarch64/matmul/brgemm_matmul_utils.cpp` choose macroblocking and
  derive tails, strides, chunks, and packing buffers.
- `src/cpu/x64/matmul/brgemm_matmul.cpp` and its AArch64 counterpart construct
  the normal full and tail BRGEMM descriptors and register scratchpad.
- x64 BRGEMM has blocking-hint machinery (including the AMX path), but the
  vector blocking path in this revision does not realize `hint_bd_block` and
  `hint_ld_block2`. AArch64 declares the same fields and honors loop order, but
  likewise does not apply those block hints in its general heuristic. The MVP
  validates dimensions and register pressure before applying the two hints in
  both vector backends.
- A blocked weights descriptor is persistent B packing: the application
  reorders B once into the primitive descriptor's selected weights format.
  `use_buffer_b` is different: it is execution scratchpad populated by the
  MatMul copy kernel on every call.
- The global primitive cache key does not include a tuning plan. `matopt`
  therefore sets the cache capacity to zero before creating primitives.

## Build and internal boundary

`DNNL_BUILD_MATOPT=ON` builds a non-installed `matopt` executable and internal
adapter. It requires:

```text
DNNL_LIBRARY_TYPE=STATIC
DNNL_CPU_RUNTIME=OMP
```

The option does not add symbols to the installed public C or C++ API. The
normal null-plan path does not execute plan logic. When enabled, a scoped,
thread-local internal plan is visible only while the tuner creates a primitive
descriptor and its existing kernel bundle.

The descriptor setup sequence is:

```text
normalize ordinary oneDNN MatMul descriptors
    -> run the existing architecture heuristic
    -> capture it, or validate and apply explicit macro overrides
    -> validate packing/layout relationships
    -> run the existing tail/stride/chunk/buffer finalization
    -> create the existing full and tail BRGEMM descriptors
    -> apply micro hints to the full M/N/K descriptor only
```

Invalid explicit plans return an error. They never silently fall back to the
heuristic.

## Versioned records

`workload_t` records schema version, M/N/K, FP32 dense layout invariants,
`alpha=1`, `beta=0`, thread count, exact CPU mask, and requested objectives.

`plan_t` contains only MVP choices:

- `M_blk`, `N_blk`, `K_blk`;
- M/N/K chunk sizes, BRGEMM batch size, and `nthr_k`;
- A packing policy;
- B packing policy and implied blocked-N layout;
- full-descriptor `bd_block`, `ld_block2`, and loop order.

`finalized_plan_t` captures realized M/N/K, tails, strides, chunk counts,
packing flags, blocked-B state, scratchpad bytes, and realized microtiles.
Serialization is schema-versioned and stable hashes use canonical field order.

Supported A policies are `direct` and `per_call_padded`. Supported B policies
are `direct`, `per_call_n32`, `per_call_n64`, `persistent_n32`, and
`persistent_n64`. An n32/n64 policy is invalid unless `N_blk` agrees with the
format. Persistent policies require a blocked weights descriptor; per-call
policies require `use_buffer_b`.

The upstream AArch64 plain copy-A JIT in this revision is unfinished. For the
MVP's dense FP32 `per_call_padded` policy, the patch therefore uses a bounded
native row-copy kernel with the finalized padded `LDA` and zero-filled row
padding. Other AArch64 copy-A domains remain rejected or follow their existing
oneDNN path; this fallback can be replaced by an optimized SVE copy kernel
without changing the plan or runner protocol.

This capability change uses MatOpt patch-set fingerprint `external-v2` so
histories containing terminal padded-A failures from `external-v1` cannot
silently suppress reevaluation.

## Search

The tuner first creates the ordinary primitive, captures its configuration,
replays that configuration through the explicit-plan path, validates the
result, and benchmarks it. This baseline is never inferred from hard-coded
defaults.

See [search_space.md](search_space.md) for the detailed hierarchy of thread,
cache, scheduler, packing, and microkernel controls; conditional constraints;
current range rationale; and the proposed configurable-space extension.

Macro candidates use:

```text
M_blk: baseline, 64, 96, 128, 160, 192, 224, 256
N_blk: baseline, 32, 64
K_blk: baseline, 256, 512, 1024
N_chunk_size: baseline, 1, 2, 4, 8
BRGEMM batch size: baseline, 1, 2, 4
nthr_k: valid divisors among baseline, 1, 2, 4

For the current AArch64 MVP, split-K overrides are disabled: candidate
generation fixes `nthr_k=1`, and the backend rejects explicit plans with
`nthr_k>1`. This is a temporary safety restriction until the AArch64 split-K
execution path is validated under repeated multithreaded tuning.
A policy: all supported values
B policy: all supported values
```

Candidates are rejected for non-positive or oversized blocks, K granularity,
incompatible B layout, invalid K-thread divisors, more K threads than realized
K chunks, a BRGEMM batch larger than the available K blocks, invalid microtiles, or an
estimated scratchpad footprint above 64 MiB per thread. Remaining candidates
are ranked by baseline distance, packing footprint, tail waste, and stable
candidate hash, then capped at 256. Execution order is a deterministic shuffle
derived from the resume key.

The top three macro results per objective seed a micro search over:

```text
bd_block: baseline, 4, 5, 6, 7
ld_block2: baseline, 2, 3, 4
loop order: default, one-load
```

Hints are applied only to a descriptor with full M, N, K and full BRGEMM batch.
Tail descriptors retain oneDNN's existing heuristic.

## Correctness and measurement

Every timed candidate first runs against output from the default oneDNN
primitive. Inputs are deterministic, predominantly small integer-valued FP32,
with reproducible non-integral samples. A wrong candidate is logged and never
ranked.

Allocation and primitive/JIT creation are outside all timings:

- one-shot latency includes persistent B reorder and every internal per-call
  A/B copy;
- steady-state latency performs persistent B reorder once before timing;
- throughput repeatedly executes one MatMul at the requested intra-op thread
  count and reports sustained GFLOP/s.

There are three warmups. Macro rounds use three samples with at least 100 ms
per sample, micro rounds use five samples with at least 200 ms, and finalists
use fifteen samples with at least one second. Records contain median, minimum,
p90, GFLOP/s, packing time, packed bytes and GB/s, compute time, correctness,
stability, and scratchpad bytes.

A finalist with `(p90-median)/median > 5%` is rerun once and cannot win if it
remains unstable. Selection computes a Pareto set over one-shot latency,
steady-state latency, throughput, and scratchpad. For each requested objective,
the baseline wins unless improvement exceeds both 1% and measured baseline
noise.

## CLI, affinity, and persistence

```text
matopt tune --m M --n N --k K --threads T --cpus MASK \
  --objectives one-shot,steady-state,throughput --history run.jsonl \
  --output plans.json

matopt benchmark --plan plan.json --m M --n N --k K --threads T \
  --cpus MASK --compare-default

matopt inspect --plan plan.json
```

The CPU mask is mandatory. `matopt` applies process affinity before creating an
engine or initializing timed work, requests close/core OpenMP placement, and
checks every observed worker CPU is in the mask.

History is append-only JSONL and writes a `started` record before primitive
creation or timing, followed by rejected, incorrect, or benchmarked outcomes.
This preserves the exact candidate if a backend process terminates. A truncated unterminated final line is ignored; any
other malformed complete line is an error. Resume keys include workload,
schema, CPU model, effective ISA/vector width, cache sizes, thread count, exact
mask, oneDNN hash, and compiler identity. A mismatch rejects the history.
Selected plans and the Pareto set are written atomically as compact JSON.

## Verification and acceptance

Platform-independent tests cover schema round trips, stable hashes, candidate
generation and cap, invalid-plan reasons, scratch estimates, JSONL recovery,
Pareto selection, and baseline fallback. Backend tests must additionally cover
capture/replay, every packing policy, combined M/N/K tails, invalid microtiles,
scratchpad rejection, and realized descriptor hints.

The existing oneDNN MatMul and BRGEMM suites must pass with a null plan. Target
host runs are required on both SVE256 and AVX-512 for 4096 cubed, a combined
tail, tall-skinny, short-wide, and large-K workload, each at one thread and a
pinned multi-thread configuration.

Acceptance is correctness-based: captured/replayed configuration fields match,
every benchmarked candidate is numerically correct, tuning resumes safely,
noise causes baseline fallback, and both target ISAs execute successfully.
Performance improvement is reported, not required.

## Future external tuner

The follow-on architecture moves search policy and persistence out of the
oneDNN tree while retaining a minimal native evaluator around this internal
configuration seam. See [external_tuner_design.md](external_tuner_design.md)
for the process protocol, Python/LFBO boundary, migration stages, and end-user
patch packaging model.
