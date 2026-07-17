from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


class BenchmarkError(ValueError):
    pass


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise BenchmarkError("cannot summarize an empty sample")
    if not 0 <= q <= 1:
        raise BenchmarkError("percentile must be in [0, 1]")
    # Nearest-rank is deterministic and matches the tuner's p90 convention.
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def stats(values: Iterable[float], flops: float) -> dict[str, float]:
    samples = sorted(float(value) for value in values)
    if not samples or any(not math.isfinite(v) or v <= 0 for v in samples):
        raise BenchmarkError("latencies must be finite and positive")
    median = samples[len(samples) // 2]
    return {
        "minimum_ms": samples[0],
        "median_ms": median,
        "p90_ms": percentile(samples, 0.9),
        "gflops": flops / (median * 1.0e6),
    }


def make_summary(
    manifest: Mapping[str, Any],
    cases: Mapping[str, Iterable[float]],
    *,
    affinity: str,
    openblas: Mapping[str, Any],
    warnings: Iterable[str] = (),
) -> dict[str, Any]:
    workload = manifest.get("workload")
    if not isinstance(workload, Mapping):
        raise BenchmarkError("manifest has no workload")
    try:
        flops = 2.0 * int(workload["m"]) * int(workload["n"]) * int(workload["k"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkError("manifest workload is malformed") from exc
    required = {
        "MatOpt/create",
        "MatOpt/prepare_weights",
        "MatOpt/one_shot",
        "MatOpt/steady_throughput",
        "OpenBLAS/sgemm",
    }
    missing = required.difference(cases)
    if missing:
        raise BenchmarkError(f"missing benchmark cases: {', '.join(sorted(missing))}")
    summarized = {name: stats(values, flops) for name, values in cases.items()}
    one = summarized["MatOpt/one_shot"]
    steady = summarized["MatOpt/steady_throughput"]
    blas = summarized["OpenBLAS/sgemm"]
    return {
        "schema_version": 1,
        "kernel_id": manifest.get("kernel_id"),
        "objective": manifest.get("objective"),
        "selected_objective_result": manifest.get("selected_measurement"),
        "cases": summarized,
        "ratios": {
            "one_shot_latency_vs_openblas": one["median_ms"] / blas["median_ms"],
            "steady_latency_vs_openblas": steady["median_ms"] / blas["median_ms"],
            "steady_gflops_vs_openblas": steady["gflops"] / blas["gflops"],
        },
        "preparation_cost_ms": summarized["MatOpt/prepare_weights"]["median_ms"],
        "runtime_compatibility_warnings": list(warnings),
        "cpu_affinity": affinity,
        "openblas": dict(openblas),
    }
