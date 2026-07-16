from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


METRICS = {
    "one_shot": ("one_shot_ms", "One-shot latency (ms)"),
    "steady": ("steady_ms", "Steady-state latency (ms)"),
    "median": ("median_ms", "Median latency (ms)"),
}


@dataclass(frozen=True)
class TrajectoryPoint:
    evaluation: int
    timestamp_ns: int
    generation: int
    latency_ms: float
    phase: str
    plan_hash: str


def load_records(path: str | Path) -> List[Dict[str, Any]]:
    """Load a history without requiring the caller to know its fingerprint."""
    source = Path(path)
    raw = source.read_bytes()
    lines = raw.splitlines(keepends=True)
    records: List[Dict[str, Any]] = []
    fingerprints = set()
    for index, line in enumerate(lines):
        complete = line.endswith(b"\n") or line.endswith(b"\r")
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index == len(lines) - 1 and not complete:
                break
            raise ValueError(f"malformed complete JSONL record {index + 1}") from exc
        fingerprint = record.get("fingerprint")
        if not fingerprint:
            raise ValueError(f"history record {index + 1} has no fingerprint")
        fingerprints.add(str(fingerprint))
        records.append(record)
    if len(fingerprints) > 1:
        raise ValueError("history contains multiple fingerprints")
    return records


def trajectory_points(
    records: Iterable[Dict[str, Any]], metric: str = "one_shot"
) -> List[TrajectoryPoint]:
    if metric not in METRICS:
        raise ValueError(f"unknown latency metric: {metric}")
    measurement_key = METRICS[metric][0]
    terminal = []
    for order, record in enumerate(records):
        if record.get("state") != "benchmarked":
            continue
        measurement = record.get("response", {}).get("measurement", {})
        if not measurement.get("correct", False):
            continue
        try:
            latency = float(measurement[measurement_key])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(latency) or latency <= 0:
            continue
        terminal.append((int(record.get("timestamp_ns", 0)), order, record, latency))

    terminal.sort(key=lambda item: (item[0], item[1]))
    points = []
    for evaluation, (timestamp, _, record, latency) in enumerate(terminal):
        search = record.get("search", {})
        generation = -1 if record.get("phase") == "baseline" else int(
            search.get("generation", 0)
        )
        points.append(
            TrajectoryPoint(
                evaluation=evaluation,
                timestamp_ns=timestamp,
                generation=generation,
                latency_ms=latency,
                phase=str(record.get("phase", "search")),
                plan_hash=str(record.get("plan_hash", "")),
            )
        )
    return points


def pareto_steps(
    points: Sequence[TrajectoryPoint],
) -> List[Tuple[int, int, float]]:
    """Return generation endpoint, generation, and cumulative minimum latency."""
    if not points:
        return []
    steps: List[Tuple[int, int, float]] = []
    best = math.inf
    current_generation = points[0].generation
    endpoint = points[0].evaluation
    for point in points:
        if point.generation != current_generation:
            steps.append((endpoint, current_generation, best))
            current_generation = point.generation
        best = min(best, point.latency_ms)
        endpoint = point.evaluation
    steps.append((endpoint, current_generation, best))
    return steps


def plot_trajectory(
    history: str | Path,
    output: str | Path,
    *,
    metric: str = "one_shot",
    title: str | None = None,
    dpi: int = 160,
) -> Path:
    config_dir = Path(tempfile.gettempdir()) / "matopt-matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "trajectory plotting requires matplotlib; install "
            "matopt-tuner[visualization] or run `uv sync --extra visualization`"
        ) from exc

    if dpi <= 0:
        raise ValueError("dpi must be positive")
    points = trajectory_points(load_records(history), metric)
    if not points:
        raise ValueError("history has no valid benchmarked latency records")
    steps = pareto_steps(points)

    figure, axis = plt.subplots(figsize=(10.5, 6.0), constrained_layout=True)
    search_points = [point for point in points if point.generation >= 0]
    baseline_points = [point for point in points if point.generation < 0]
    if search_points:
        scatter = axis.scatter(
            [point.evaluation for point in search_points],
            [point.latency_ms for point in search_points],
            c=[point.generation for point in search_points],
            cmap="viridis",
            alpha=0.78,
            edgecolors="none",
            s=34,
            label="Benchmarked candidate",
            zorder=2,
        )
        colorbar = figure.colorbar(scatter, ax=axis, pad=0.02)
        colorbar.set_label("LFBO generation")
    if baseline_points:
        axis.scatter(
            [point.evaluation for point in baseline_points],
            [point.latency_ms for point in baseline_points],
            marker="*",
            color="#d62728",
            edgecolors="black",
            linewidths=0.5,
            s=145,
            label="oneDNN baseline",
            zorder=4,
        )

    step_x = [step[0] for step in steps]
    step_y = [step[2] for step in steps]
    axis.step(
        step_x,
        step_y,
        where="post",
        color="#111111",
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="Current Pareto optimum by generation",
        zorder=3,
    )
    for endpoint, generation, latency in steps:
        label = "baseline" if generation < 0 else f"g{generation}"
        axis.annotate(
            label,
            (endpoint, latency),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=8,
            color="#333333",
        )

    axis.set_xlabel("Evaluation timeline")
    axis.set_ylabel(METRICS[metric][1])
    axis.set_title(title or "MatOpt LFBO optimization trajectory")
    axis.grid(True, which="major", alpha=0.22)
    axis.set_xlim(-0.5, max(point.evaluation for point in points) + 0.5)
    axis.legend(loc="best")

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi)
    plt.close(figure)
    return destination
