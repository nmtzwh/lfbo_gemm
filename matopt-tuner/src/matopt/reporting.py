from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, TextIO


PLAN_FIELDS = (
    ("M_blk", "M"),
    ("N_blk", "N"),
    ("K_blk", "K"),
    ("M_chunk_size", "mc"),
    ("N_chunk_size", "nc"),
    ("brgemm_batch_size", "bs"),
    ("nthr_k", "tk"),
    ("pack_a", "A"),
    ("pack_b", "B"),
    ("bd_block", "bd"),
    ("ld_block2", "ld2"),
    ("loop_order", "loop"),
)


class ConsoleReporter:
    """Compact colored tuning progress written to a diagnostic stream."""

    _codes = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "cyan": "\033[36m",
    }

    def __init__(
        self,
        level: int = 1,
        *,
        stream: TextIO | None = None,
        color: str = "auto",
    ) -> None:
        if level < 0:
            raise ValueError("verbosity level must be non-negative")
        if color not in {"auto", "always", "never"}:
            raise ValueError("color must be auto, always, or never")
        self.level = level
        self.stream = stream or sys.stderr
        self.use_color = color == "always" or (
            color == "auto"
            and "NO_COLOR" not in os.environ
            and bool(getattr(self.stream, "isatty", lambda: False)())
        )
        self.objective = "one_shot"
        self.baseline: Dict[str, Any] | None = None
        self.best: Dict[str, Any] | None = None
        self.generation: int | None = None
        self.generation_records: List[Dict[str, Any]] = []
        self.evaluation = 0

    def __call__(self, event: str, payload: Dict[str, Any]) -> None:
        if self.level == 0:
            return
        if event == "start":
            self._start(payload)
        elif event == "baseline":
            self._baseline(payload["record"], payload.get("prior_records", []))
        elif event == "candidate":
            self._candidate(payload["record"])
        elif event == "finish":
            self._finish(payload["result"])

    def _style(self, value: str, *styles: str) -> str:
        if not self.use_color:
            return value
        prefix = "".join(self._codes[style] for style in styles)
        return f"{prefix}{value}{self._codes['reset']}"

    def _write(self, value: str) -> None:
        print(value, file=self.stream, flush=True)

    def _label(self, value: str) -> str:
        return self._style(value, "bold", "cyan")

    @staticmethod
    def _domain(values: List[Any]) -> str:
        return ",".join(str(value) for value in values)

    def _start(self, payload: Dict[str, Any]) -> None:
        workload = payload["workload"]
        caps = payload["capabilities"]
        self.objective = payload["objective"]
        algorithm = payload["algorithm"].upper()
        problem = (
            f"{workload.m}x{workload.n}x{workload.k} {workload.dtype} "
            f"threads={workload.threads} cpus={workload.cpus}"
        )
        isa = caps.get("effective_isa", "unknown")
        resumed = int(payload.get("resumed", 0))
        suffix = f" resumed={resumed}" if resumed else ""
        self._write(
            f"{self._label('MATOPT')} {self._style(problem, 'bold')} | "
            f"isa={isa} search={algorithm} objective={self.objective} "
            f"budget={payload['budget']}{suffix}"
        )
        domains = payload["space"]["effective_domains"]
        limits = payload["space"]["limits"]
        self._write(
            f"{self._label('SPACE')} blocks "
            f"M={self._domain(domains['M_blk'])} "
            f"N={self._domain(domains['N_blk'])} "
            f"K={self._domain(domains['K_blk'])} | chunks "
            f"M={self._domain(domains['M_chunk_size'])} "
            f"N={self._domain(domains['N_chunk_size'])} "
            f"batch={self._domain(domains['brgemm_batch_size'])} "
            f"nthr_k={self._domain(domains['nthr_k'])}"
        )
        self._write(
            f"{self._label('SPACE')} pack "
            f"A={self._domain(domains['pack_a'])} "
            f"B={self._domain(domains['pack_b'])} | micro "
            f"bd={self._domain(domains['bd_block'])} "
            f"ld2={self._domain(domains['ld_block2'])} "
            f"loop={self._domain(domains['loop_order'])} | limits "
            f"scratch={limits['scratchpad_per_thread_bytes'] / 1048576:.4g}MiB/thr "
            f"work/thr>={limits['minimum_parallel_work_per_thread']:.4g}"
        )

    @staticmethod
    def _measurement(record: Dict[str, Any]) -> Dict[str, Any] | None:
        value = record.get("response", {}).get("measurement")
        return value if isinstance(value, dict) else None

    def _objective_value(self, record: Dict[str, Any]) -> float | None:
        measurement = self._measurement(record)
        if measurement is None:
            return None
        key = {
            "one_shot": "one_shot_ms",
            "steady": "steady_ms",
            "throughput": "throughput_gflops",
        }[self.objective]
        value = measurement.get(key)
        return float(value) if value is not None else None

    def _better(self, record: Dict[str, Any], current: Dict[str, Any]) -> bool:
        value, best = self._objective_value(record), self._objective_value(current)
        if value is None:
            return False
        if best is None:
            return True
        return value > best if self.objective == "throughput" else value < best

    @staticmethod
    def _eligible(record: Dict[str, Any]) -> bool:
        measurement = ConsoleReporter._measurement(record)
        return bool(
            record.get("state") == "benchmarked"
            and measurement
            and measurement.get("correct")
            and measurement.get("stable")
        )

    def _value(self, record: Dict[str, Any]) -> str:
        value = self._objective_value(record)
        if value is None:
            return "n/a"
        unit = "GF/s" if self.objective == "throughput" else "ms"
        return f"{value:.4g} {unit}"

    @staticmethod
    def _metrics(record: Dict[str, Any]) -> str:
        measurement = ConsoleReporter._measurement(record)
        if measurement is None:
            return "no measurement"
        return (
            f"one={float(measurement['one_shot_ms']):.4g}ms "
            f"steady={float(measurement['steady_ms']):.4g}ms "
            f"perf={float(measurement['throughput_gflops']):.4g}GF/s"
        )

    @staticmethod
    def _plan(record: Dict[str, Any]) -> str:
        plan = record["plan"]
        return " ".join(
            f"{short}={plan[field]}"
            for field, short in PLAN_FIELDS
            if field in plan
        )

    def _baseline(
        self, record: Dict[str, Any], prior_records: List[Dict[str, Any]]
    ) -> None:
        self.baseline = record
        self.best = record
        for prior in prior_records:
            if self._eligible(prior) and self._better(prior, self.best):
                self.best = prior
        self._write(
            f"{self._label('BASE')} "
            f"{self._style(self._metrics(record), 'bold')} | {self._plan(record)}"
        )

    @staticmethod
    def _record_generation(record: Dict[str, Any]) -> int:
        return int(record.get("search", {}).get("generation", 0))

    def _candidate(self, record: Dict[str, Any]) -> None:
        generation = self._record_generation(record)
        if self.generation is not None and generation != self.generation:
            self._flush_generation()
        self.generation = generation
        self.generation_records.append(record)
        self.evaluation += 1
        if self._eligible(record) and self.best is not None:
            if self._better(record, self.best):
                self.best = record
        if self.level >= 2:
            self._write(self._candidate_line(record, generation))

    def _candidate_line(self, record: Dict[str, Any], generation: int) -> str:
        state = str(record.get("state", "unknown"))
        prefix = f"  #{self.evaluation:03d} g{generation}"
        if state == "benchmarked" and self._eligible(record):
            status = self._style("OK", "green", "bold")
            detail = self._metrics(record)
        elif state == "benchmarked":
            status = self._style("UNSTABLE", "yellow", "bold")
            detail = self._metrics(record)
        else:
            style = "yellow" if state == "rejected" else "red"
            status = self._style(state.upper(), style, "bold")
            response = record.get("response", {})
            reason = str(response.get("reason_code") or "")
            explanation = str(response.get("detail") or "")
            detail = reason
            if explanation and explanation != reason:
                detail = f"{reason}: {explanation}" if reason else explanation
        return f"{prefix} {status} {detail} | {self._plan(record)}"

    def _gain(self, record: Dict[str, Any]) -> float:
        if self.baseline is None:
            return 0.0
        base = self._objective_value(self.baseline)
        value = self._objective_value(record)
        if base is None or value is None or base == 0:
            return 0.0
        if self.objective == "throughput":
            return value / base - 1.0
        return 1.0 - value / base

    def _flush_generation(self) -> None:
        if self.generation is None or not self.generation_records:
            return
        valid = [
            record
            for record in self.generation_records
            if self._eligible(record)
        ]
        if valid:
            values = [self._objective_value(record) for record in valid]
            finite = [value for value in values if value is not None]
            spread = (
                f" range={min(finite):.4g}..{max(finite):.4g}"
                f"{'GF/s' if self.objective == 'throughput' else 'ms'}"
                if finite
                else ""
            )
        else:
            spread = ""
        best = self.best
        best_text = self._value(best) if best is not None else "n/a"
        gain = self._gain(best) if best is not None else 0.0
        gain_text = self._style(f"{gain:+.2%}", "green" if gain > 0 else "yellow")
        rejected = len(self.generation_records) - len(valid)
        line = (
            f"{self._label(f'GEN {self.generation}')} "
            f"ok={len(valid)} rejected={rejected}{spread} | "
            f"best={self._style(best_text, 'bold')} gain={gain_text}"
        )
        if best is not None:
            line += f" | {self._plan(best)}"
        self._write(line)
        self.generation_records.clear()

    def _finish(self, result: Dict[str, Any]) -> None:
        self._flush_generation()
        selected = result["selected"][self.objective]
        is_baseline = self.baseline is not None and (
            selected.get("plan_hash") == self.baseline.get("plan_hash")
        )
        origin = "baseline" if is_baseline else "tuned"
        gain = self._gain(selected)
        self._write(
            f"{self._label('SELECT')} {self.objective} "
            f"{self._style(self._value(selected), 'green', 'bold')} "
            f"gain={self._style(f'{gain:+.2%}', 'green' if gain > 0 else 'yellow')} "
            f"source={origin} pareto={len(result['pareto'])} | {self._plan(selected)}"
        )
