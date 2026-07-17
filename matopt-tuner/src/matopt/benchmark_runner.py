from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .benchmarking import BenchmarkError, make_summary
from .protocol import canonical_json


def _cpu_count(mask: str) -> int:
    cpus: set[int] = set()
    try:
        for part in mask.split(","):
            bounds = [int(value) for value in part.split("-")]
            if len(bounds) == 1:
                cpus.add(bounds[0])
            elif len(bounds) == 2 and bounds[0] <= bounds[1]:
                cpus.update(range(bounds[0], bounds[1] + 1))
            else:
                raise ValueError
    except ValueError as exc:
        raise BenchmarkError(f"invalid CPU mask: {mask}") from exc
    if not cpus:
        raise BenchmarkError("CPU mask is empty")
    return len(cpus)


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(command, text=True, capture_output=True, env=env)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise BenchmarkError(f"command failed ({command[0]}): {detail}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError(f"{label} must be an object")
    return value


def _cases(report: dict[str, Any]) -> dict[str, list[float]]:
    expected = (
        "MatOpt/create",
        "MatOpt/prepare_weights",
        "MatOpt/one_shot",
        "MatOpt/steady_throughput",
        "OpenBLAS/sgemm",
    )
    result = {name: [] for name in expected}
    rows = report.get("benchmarks")
    if not isinstance(rows, list):
        raise BenchmarkError("Google Benchmark report has no benchmarks")
    scales = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("run_type", "iteration") != "iteration"
        ):
            continue
        name = str(row.get("name", ""))
        base = next((case for case in expected if name.startswith(case)), None)
        if base is None:
            continue
        unit = row.get("time_unit")
        value = row.get("real_time")
        if unit not in scales or not isinstance(value, (int, float)):
            raise BenchmarkError(f"malformed timing row: {name}")
        result[base].append(float(value) * scales[unit])
    return result


def benchmark_package(
    *,
    kernel: str | os.PathLike[str],
    cpus: str,
    build_dir: str | os.PathLike[str],
    output: str | os.PathLike[str],
    fetch_google_benchmark: bool = False,
) -> dict[str, Any]:
    package = Path(kernel).resolve()
    manifest_path = package / "share/matopt/manifest.json"
    manifest = _load_json(manifest_path, "kernel manifest")
    kid = manifest.get("kernel_id")
    if not isinstance(kid, str) or not kid:
        raise BenchmarkError("kernel manifest has no ID")
    threads = manifest.get("workload", {}).get("threads")
    if _cpu_count(cpus) != threads:
        raise BenchmarkError(
            "affinity cardinality must equal the fixed kernel thread count"
        )
    if not (package / "include" / f"matopt_kernel_{kid}.hpp").is_file():
        raise BenchmarkError("generated kernel header is missing")
    build = Path(build_dir).resolve()
    build.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[3] / "benchmark"
    configure = [
        "cmake",
        "-S",
        str(project),
        "-B",
        str(build),
        f"-DMATOPT_PACKAGE={package}",
        f"-DMATOPT_KERNEL_ID={kid}",
        f"-DMATOPT_FETCH_GOOGLE_BENCHMARK={'ON' if fetch_google_benchmark else 'OFF'}",
    ]
    _run(configure)
    _run(["cmake", "--build", str(build), "--target", "matopt-openblas-benchmark"])
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        correctness = stage / "correctness.json"
        google = stage / "google-benchmark.json"
        executable = build / "matopt-openblas-benchmark"
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        _run(
            [
                "taskset",
                "-c",
                cpus,
                str(executable),
                f"--correctness={correctness}",
                f"--benchmark_out={google}",
                "--benchmark_out_format=json",
                "--benchmark_repetitions=15",
                "--benchmark_min_time=1s",
                "--benchmark_report_aggregates_only=false",
            ],
            env=env,
        )
        correctness_value = _load_json(correctness, "correctness report")
        if (
            correctness_value.get("correct") is not True
            or correctness_value.get("jit_events") != 0
        ):
            raise BenchmarkError("correctness preflight did not prove zero-JIT execution")
        google_value = _load_json(google, "Google Benchmark report")
        warnings = (
            ["execution identity differs from tuning identity"]
            if correctness_value.get("runtime_warning")
            else []
        )
        summary = make_summary(
            manifest,
            _cases(google_value),
            affinity=cpus,
            openblas={
                "configuration": correctness_value.get(
                    "openblas_configuration", "unknown"
                )
            },
            warnings=warnings,
        )
        (stage / "summary.json").write_text(
            canonical_json(summary) + "\n", encoding="utf-8"
        )
        if destination.exists():
            raise BenchmarkError(f"output already exists: {destination}")
        os.replace(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {"output": str(destination), "kernel_id": kid, "summary": summary}
