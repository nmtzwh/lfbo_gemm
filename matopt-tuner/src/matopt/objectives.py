from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _measurement(record: Dict[str, Any]) -> Dict[str, Any]:
    return record["response"]["measurement"]


def pareto(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = [
        r
        for r in records
        if r.get("state") == "benchmarked"
        and _measurement(r).get("correct")
        and _measurement(r).get("stable")
    ]
    result: List[Dict[str, Any]] = []
    for current in valid:
        a = _measurement(current)
        dominated = False
        for other in valid:
            if other is current:
                continue
            b = _measurement(other)
            no_worse = (
                b["one_shot_ms"] <= a["one_shot_ms"]
                and b["steady_ms"] <= a["steady_ms"]
                and b["throughput_gflops"] >= a["throughput_gflops"]
                and b["scratchpad_bytes"] <= a["scratchpad_bytes"]
            )
            better = (
                b["one_shot_ms"] < a["one_shot_ms"]
                or b["steady_ms"] < a["steady_ms"]
                or b["throughput_gflops"] > a["throughput_gflops"]
                or b["scratchpad_bytes"] < a["scratchpad_bytes"]
            )
            if no_worse and better:
                dominated = True
                break
        if not dominated:
            result.append(current)
    return result


def select(
    records: List[Dict[str, Any]],
    objective: str,
    baseline: Dict[str, Any],
    *,
    baseline_eligible: bool = True,
) -> Dict[str, Any]:
    valid = [
        r
        for r in records
        if r.get("state") == "benchmarked"
        and _measurement(r).get("correct")
        and _measurement(r).get("stable")
    ]
    if not valid:
        if baseline_eligible:
            return baseline
        raise ValueError("no correct stable result within configured search space")
    if objective == "throughput":
        best = max(valid, key=lambda r: _measurement(r)["throughput_gflops"])
        gain = (
            _measurement(best)["throughput_gflops"]
            / _measurement(baseline)["throughput_gflops"]
            - 1.0
        )
    else:
        key = "one_shot_ms" if objective == "one_shot" else "steady_ms"
        best = min(valid, key=lambda r: _measurement(r)[key])
        gain = 1.0 - _measurement(best)[key] / _measurement(baseline)[key]
    base_m = _measurement(baseline)
    noise = max(
        0.0,
        (base_m["p90_ms"] - base_m["median_ms"])
        / max(base_m["median_ms"], 1e-30),
    )
    return best if not baseline_eligible or gain > max(0.01, noise) else baseline
