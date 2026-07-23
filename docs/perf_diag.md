# BRGEMM performance diagnosis

`matopt-tuner diagnose` is a fail-closed, target-host diagnostic command for a
saved tuning result. It rechecks the selected record and baseline, verifies the
live runner fingerprint, and requires the selected plan to finalize exactly as
it did during tuning.

```bash
matopt-tuner diagnose \
  --result plans.json \
  --objective throughput \
  --runner /path/to/matopt-runner \
  --pmu-profile cpu-pmu.yaml \
  --nominal-peak-gflops 6400 \
  --output perf-diag-report
```

The PMU profile is never guessed. It must use schema
`perf_diag_profile_v1`, identify the architecture, CPU model, ISA and vector
width, declare homogeneous cores, specify FP32 FLOPs/cycle/core, and map every
required semantic counter role to a target-validated `perf` event. The example
in `matopt-tuner/examples/perf_diag_profile_v1.yaml` is a schema example, not a
validated profile for arbitrary x86 systems.

The runner is invoked under `perf stat --delay=-1 --control fd:...`. Every
semantic event is collected in a separate pass, so authoritative results never
depend on multiplexed counter scaling. Primitive construction, JIT,
initialization, and warmup happen before the runner enables the counter window.

The report is published by a single directory rename and contains:

- `manifest.json`: inputs, identity, and final status.
- `trace.json`: address-free logical execution trace.
- `stage-samples.json`: raw timing samples and confidence estimates.
- `pmu.json`: semantic counters and running ratios for every stage.
- `attribution.json`: signed nominal-peak and run-rate waterfall data.
- `summary.md`: compact human-readable waterfall.

An existing output directory is never merged or overwritten. Missing PMU
roles, low running ratios, CPU migrations, unstable or drifting passes, an
instrumentation perturbation above 1%, an inexact native trace, or a residual
above 20% of the nominal-peak gap makes the report `inconclusive`; raw evidence
is still published and the command exits nonzero.

The native command captures an exact address-free worker/variant/block trace.
The controlled hot/ideal/private/shared stages are currently marked
`driver_proxy`, however, and therefore deliberately prevent an authoritative
report. Target attribution requires the follow-on direct BRGEMM inventory
replay implementation; proxy timings are never presented as contributions.
