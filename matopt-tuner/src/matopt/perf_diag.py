from __future__ import annotations

import json
import math
import os
import platform
import re
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .exporter import ExportError, _workload, load_result, selected_record
from .protocol import Workload, canonical_json, stable_hash
from .runner import MatOptRunner

PROFILE_SCHEMA = "perf_diag_profile_v1"
REPORT_SCHEMA = "perf_diag_report_v1"
REQUIRED_ROLES = {
    "cycles",
    "task_clock",
    "instructions",
    "l1_accesses",
    "l1_refills",
    "l2_accesses",
    "l2_refills",
    "llc_accesses",
    "llc_refills",
    "memory_stall_cycles",
    "context_switches",
    "cpu_migrations",
}
STAGES = (
    "hot_pipeline",
    "ideal_data",
    "private_trace",
    "shared_trace",
    "full_driver",
    "empty_parallel",
    "barriers",
    "noop_chunks",
    "instrumented_driver",
    "memory_calibration",
    "cache_calibration",
)


class PerfDiagError(RuntimeError):
    pass


@dataclass(frozen=True)
class PMUProfile:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "PMUProfile":
        try:
            value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise PerfDiagError(f"cannot read PMU profile: {exc}") from exc
        if not isinstance(value, dict) or value.get("schema") != PROFILE_SCHEMA:
            raise PerfDiagError(f"PMU profile must use {PROFILE_SCHEMA}")
        for key in ("architecture", "cpu_model_regex", "isa", "vector_bits",
                    "fp32_flops_per_cycle_per_core", "events"):
            if key not in value:
                raise PerfDiagError(f"PMU profile is missing {key}")
        if value["architecture"] not in {"x86_64", "aarch64"}:
            raise PerfDiagError("unsupported profile architecture")
        if not isinstance(value["vector_bits"], int) or value["vector_bits"] <= 0:
            raise PerfDiagError("vector_bits must be positive")
        if not isinstance(value["fp32_flops_per_cycle_per_core"], (int, float)) \
                or value["fp32_flops_per_cycle_per_core"] <= 0:
            raise PerfDiagError("FP32 FLOPs/cycle/core must be positive")
        events = value["events"]
        if not isinstance(events, dict):
            raise PerfDiagError("events must be a mapping")
        missing = sorted(REQUIRED_ROLES - events.keys())
        if missing:
            raise PerfDiagError("PMU profile is missing roles: " + ", ".join(missing))
        for role, event in events.items():
            if not isinstance(role, str) or not isinstance(event, str) or not event:
                raise PerfDiagError("PMU event mappings must be non-empty strings")
        try:
            re.compile(str(value["cpu_model_regex"]))
        except re.error as exc:
            raise PerfDiagError(f"invalid cpu_model_regex: {exc}") from exc
        if value.get("homogeneous_cores") is not True:
            raise PerfDiagError("perf_diag_v1 requires homogeneous_cores: true")
        return cls(dict(value))

    def validate_host(self, capabilities: Mapping[str, Any]) -> None:
        arch = platform.machine().lower()
        aliases = {"amd64": "x86_64", "arm64": "aarch64"}
        arch = aliases.get(arch, arch)
        if arch != self.raw["architecture"]:
            raise PerfDiagError(f"profile architecture {self.raw['architecture']} "
                                f"does not match host {arch}")
        identity = str(capabilities.get("identity", ""))
        if not re.search(str(self.raw["cpu_model_regex"]), identity):
            raise PerfDiagError("CPU identity does not match cpu_model_regex")
        isa = str(capabilities.get("effective_isa", "")).lower()
        if str(self.raw["isa"]).lower() not in isa:
            raise PerfDiagError("live runner ISA does not match PMU profile")

    @property
    def events(self) -> dict[str, str]:
        return dict(self.raw["events"])


def parse_perf_csv(text: str, event_to_role: Mapping[str, str]) -> dict[str, Any]:
    counters: dict[str, Any] = {}
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(";")]
        if len(fields) < 3 or fields[0].startswith("#"):
            continue
        raw, _unit, event = fields[:3]
        role = event_to_role.get(event)
        if role is None:
            continue
        if raw in {"<not counted>", "<not supported>"}:
            counters[role] = {"available": False, "reason": raw}
            continue
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        enabled = running = None
        numeric = []
        for field in fields[3:]:
            try:
                numeric.append(float(field.replace(",", "").rstrip("%")))
            except ValueError:
                pass
        if len(numeric) >= 2:
            # perf -x emits run-time followed by the percentage running.
            enabled, running = 100.0, numeric[-1]
        ratio = running / enabled if enabled and running is not None else 1.0
        counters[role] = {
            "available": True, "value": value, "enabled": enabled,
            "running": running, "running_ratio": ratio,
        }
    return counters


def confidence(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise PerfDiagError("stage contains no samples")
    median = statistics.median(values)
    if len(values) == 1:
        return {"median": median, "low": median, "high": median, "cv": 0.0}
    stdev = statistics.stdev(values)
    half = 1.96 * stdev / math.sqrt(len(values))
    return {
        "median": median, "low": median - half, "high": median + half,
        "cv": stdev / median if median else math.inf,
    }


def attribute(*, workload: Workload, nominal_peak_gflops: float,
              fp32_flops_per_cycle: float, active_frequency_ghz: float,
              stages_ms: Mapping[str, float],
              auxiliary_ms: float, scheduling_ms: float,
              driver_control_ms: float) -> dict[str, Any]:
    flops = 2.0 * workload.m * workload.n * workload.k
    tnominal = flops / (nominal_peak_gflops * 1e6)
    trun = flops / (workload.threads * fp32_flops_per_cycle
                    * active_frequency_ghz * 1e6)
    thot = stages_ms["hot_pipeline"]
    tideal = stages_ms["ideal_data"]
    tprivate = stages_ms["private_trace"]
    tshared = stages_ms["shared_trace"]
    te2e = stages_ms["full_driver"]
    raw = [
        ("frequency_power", trun - tnominal),
        ("in_cache_pipeline", thot - trun),
        ("load_store_data_supply", tideal - thot),
        ("private_capacity_conflict", tprivate - tideal),
        ("shared_cache_contention", tshared - tprivate),
        ("packing_copy_reduction", auxiliary_ms),
        ("thread_scheduling", scheduling_ms),
        ("driver_control", driver_control_ms),
    ]
    residual = te2e - tnominal - sum(value for _, value in raw)
    raw.append(("residual", residual))
    peak_gap = te2e - tnominal
    components = []
    for name, value in raw:
        components.append({
            "name": name,
            "milliseconds": value,
            "cycles_at_run_rate": value * active_frequency_ghz * 1e6,
            "percent_te2e": 100 * value / te2e if te2e else math.nan,
            "percent_peak_gap": 100 * value / peak_gap if peak_gap else math.nan,
            "efficiency_if_removed": (
                100 * tnominal / (te2e - value)
                if te2e != value else math.inf
            ),
        })
    return {
        "flops": flops,
        "times_ms": {"nominal": tnominal, "run_rate": trun, "hot": thot,
                     "ideal": tideal, "private": tprivate, "shared": tshared,
                     "end_to_end": te2e},
        "components": components,
        "closure_ms": tnominal + sum(c["milliseconds"] for c in components),
        "peak_gap_ms": peak_gap,
        "residual_fraction_of_peak_gap": (
            abs(residual) / abs(peak_gap) if peak_gap else math.inf
        ),
    }


def _atomic_publish(output: Path, files: Mapping[str, Any]) -> None:
    output = output.resolve()
    if output.exists():
        raise PerfDiagError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.",
                                      dir=output.parent))
    try:
        for name, value in files.items():
            path = temporary / name
            if isinstance(value, str):
                path.write_text(value, encoding="utf-8")
            else:
                path.write_text(canonical_json(value) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _summary(attribution: Mapping[str, Any], status: str,
             reasons: Sequence[str]) -> str:
    lines = [f"# BRGEMM performance diagnosis", "", f"Status: **{status}**.", ""]
    if reasons:
        lines += ["Validation findings:", ""]
        lines += [f"- {reason}" for reason in reasons]
        lines.append("")
    lines += ["| Component | ms | % Te2e | % peak gap |", "|---|---:|---:|---:|"]
    for item in attribution["components"]:
        lines.append(f"| {item['name']} | {item['milliseconds']:.6f} | "
                     f"{item['percent_te2e']:.2f} | "
                     f"{item['percent_peak_gap']:.2f} |")
    lines += ["", "Negative components are retained; they indicate noise or "
              "interaction between controlled stages."]
    return "\n".join(lines) + "\n"


def diagnose(*, result_path: str | os.PathLike[str], objective: str,
             runner_path: str | os.PathLike[str],
             profile_path: str | os.PathLike[str],
             nominal_peak_gflops: float, output: str | os.PathLike[str],
             kernel: str | os.PathLike[str] | None = None,
             timeout: float = 300.0) -> dict[str, Any]:
    if nominal_peak_gflops <= 0:
        raise PerfDiagError("nominal peak must be positive")
    try:
        result = load_result(result_path)
        record = selected_record(result, objective)
        baseline = result.get("baseline")
        if not isinstance(baseline, dict):
            raise PerfDiagError("tuning result has no selected baseline")
        selected_record({"selected": {objective: baseline}}, objective)
        workload = _workload(result)
    except ExportError as exc:
        raise PerfDiagError(str(exc)) from exc
    runner = MatOptRunner(runner_path, timeout=timeout)
    caps = runner.capabilities(workload)
    if caps.get("status") != "capabilities":
        raise PerfDiagError(f"live runner capability discovery failed: {caps}")
    live = caps.get("fingerprint")
    if live != result.get("runner_fingerprint"):
        raise PerfDiagError("runner fingerprint does not match tuning result")
    inspected = runner.inspect(workload, record["plan"], str(live))
    if inspected.get("status") != "accepted":
        raise PerfDiagError(f"selected plan no longer finalizes: {inspected}")
    if inspected.get("finalized") != record["response"]["finalized"]:
        raise PerfDiagError("re-finalized plan differs from saved plan")
    profile = PMUProfile.load(profile_path)
    profile.validate_host(caps)
    if shutil.which("perf") is None:
        raise PerfDiagError("Linux perf is required")

    responses: dict[str, Any] = {}
    for stage in STAGES:
        response = runner.perf_diagnose_stage(
            workload, record["plan"], str(live), stage, profile.events)
        if response.get("status") != "diagnosed":
            raise PerfDiagError(f"native stage {stage} failed: {response}")
        responses[stage] = response
    trace = responses["full_driver"].get("trace")
    if not isinstance(trace, dict) or trace.get("flops") != 2 * workload.m * workload.n * workload.k:
        raise PerfDiagError("native trace FLOPs do not equal 2MNK")
    stage_samples = {
        stage: response.get("samples_ms", []) for stage, response in responses.items()
    }
    estimates = {stage: confidence([float(x) for x in samples])
                 for stage, samples in stage_samples.items()}
    pmu = {stage: response.get("pmu", {}) for stage, response in responses.items()}
    full_pmu = pmu["full_driver"]
    cycles = float(full_pmu.get("cycles", {}).get("value", 0))
    task_ms = float(full_pmu.get("task_clock", {}).get("value", 0))
    active_frequency = cycles / (task_ms * 1e6) if task_ms > 0 else 0.0
    reasons = []
    for stage, counters in pmu.items():
        missing = REQUIRED_ROLES - counters.keys()
        if missing:
            reasons.append(f"{stage}: missing PMU roles {sorted(missing)}")
        for role, counter in counters.items():
            if role in REQUIRED_ROLES and (
                    not counter.get("available")
                    or float(counter.get("running_ratio", 0)) < 0.95):
                reasons.append(f"{stage}: unusable {role} counter")
        drift = float(responses[stage].get("counter_pass_timing_drift", 0))
        if drift > 0.05:
            reasons.append(f"{stage}: counter-pass timing drift exceeds 5%")
    migrations = full_pmu.get("cpu_migrations", {}).get("value")
    if migrations != 0:
        reasons.append("CPU migrations were observed")
    if not active_frequency:
        reasons.append("active frequency could not be derived")
        active_frequency = 1.0
    for stage, estimate in estimates.items():
        if estimate["cv"] > 0.05:
            reasons.append(f"{stage}: unstable timing samples")
    for stage in ("hot_pipeline", "ideal_data", "private_trace", "shared_trace"):
        if responses[stage].get("stage_fidelity") != "native":
            reasons.append(f"{stage}: native replay stage is unavailable")
    full = responses["full_driver"]
    attribution = attribute(
        workload=workload, nominal_peak_gflops=nominal_peak_gflops,
        fp32_flops_per_cycle=float(profile.raw["fp32_flops_per_cycle_per_core"]),
        active_frequency_ghz=active_frequency,
        stages_ms={name: estimates[name]["median"] for name in (
            "hot_pipeline", "ideal_data", "private_trace", "shared_trace",
            "full_driver")},
        auxiliary_ms=float(full.get("auxiliary_ms", 0)),
        scheduling_ms=float(full.get("scheduling_ms", 0)),
        driver_control_ms=float(full.get("driver_control_ms", 0)),
    )
    if attribution["residual_fraction_of_peak_gap"] > 0.20:
        reasons.append("absolute residual exceeds 20% of the nominal-peak gap")
    instrumentation_delta = float(
        responses["instrumented_driver"].get("instrumentation_delta", 0))
    if abs(instrumentation_delta) > 0.01:
        reasons.append("instrumentation changes end-to-end latency by more than 1%")
    if trace.get("fidelity") != "exact_worker_trace":
        reasons.append("native runner did not provide an exact worker trace")
    kernel_cross_check = None
    if kernel:
        package = Path(kernel).resolve()
        manifest_path = package / "share/matopt/manifest.json"
        try:
            package_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PerfDiagError(f"cannot read kernel manifest: {exc}") from exc
        if package_manifest.get("workload") != workload.to_dict():
            raise PerfDiagError("kernel package workload does not match result")
        if package_manifest.get("finalized_plan") != record["response"]["finalized"]:
            raise PerfDiagError("kernel package finalized plan does not match result")
        validation = runner.validate_aot(workload, str(live), package)
        if validation.get("status") != "validated" or validation.get("jit_events") != 0:
            raise PerfDiagError(f"kernel package validation failed: {validation}")
        kernel_cross_check = {
            "package": str(package), "correct": True, "jit_events": 0,
            "steady_throughput_check": "unavailable",
        }
        reasons.append("kernel steady-throughput cross-check was not measured")
    status = "authoritative" if not reasons else "inconclusive"
    attribution["status"] = status
    attribution["reasons"] = reasons
    manifest = {
        "schema": REPORT_SCHEMA, "status": status, "created_unix_ns": time.time_ns(),
        "objective": objective, "runner_fingerprint": live,
        "workload": workload.to_dict(), "nominal_peak_gflops": nominal_peak_gflops,
        "pmu_profile": profile.raw, "kernel_cross_check": kernel_cross_check,
        "trace_hash": stable_hash(trace),
    }
    _atomic_publish(Path(output), {
        "manifest.json": manifest, "trace.json": trace,
        "stage-samples.json": {"samples_ms": stage_samples, "confidence": estimates},
        "pmu.json": pmu, "attribution.json": attribution,
        "summary.md": _summary(attribution, status, reasons),
    })
    return {"output": str(Path(output).resolve()), "status": status,
            "reasons": reasons}
