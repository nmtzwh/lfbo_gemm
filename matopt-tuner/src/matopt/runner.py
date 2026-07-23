from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping

from .protocol import MeasurementProfile, Workload, canonical_json, request, validate_response


class MatOptRunner:
    def __init__(
        self,
        executable: str | os.PathLike[str],
        *,
        timeout: float = 300.0,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.executable = str(Path(executable).resolve())
        self.timeout = timeout
        self.env = dict(env or {})
        if not os.path.isfile(self.executable):
            raise FileNotFoundError(self.executable)

    def _run(
        self,
        command: str,
        payload: Dict[str, Any],
        *extra_args: str,
    ) -> Dict[str, Any]:
        request_id = str(payload["request_id"])
        path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".json", delete=False
            ) as stream:
                path = stream.name
                stream.write(canonical_json(payload))
                stream.write("\n")
            environment = os.environ.copy()
            environment.update(self.env)
            completed = subprocess.run(
                [self.executable, command, "--request", path, *extra_args],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                env=environment,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "protocol_version": 1,
                "request_id": request_id,
                "status": "timed_out",
                "detail": str(exc),
            }
        finally:
            if path:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
        if completed.returncode < 0:
            sig = -completed.returncode
            return {
                "protocol_version": 1,
                "request_id": request_id,
                "status": "crashed",
                "signal": sig,
                "signal_name": signal.Signals(sig).name,
                "stderr": completed.stderr,
            }
        if completed.returncode != 0:
            return {
                "protocol_version": 1,
                "request_id": request_id,
                "status": "runner_error",
                "exit_code": completed.returncode,
                "stderr": completed.stderr,
            }
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            return {
                "protocol_version": 1,
                "request_id": request_id,
                "status": "protocol_error",
                "detail": "runner stdout must contain exactly one JSON object",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        try:
            return validate_response(json.loads(lines[0]), request_id)
        except (json.JSONDecodeError, ValueError) as exc:
            return {
                "protocol_version": 1,
                "request_id": request_id,
                "status": "protocol_error",
                "detail": str(exc),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }

    @staticmethod
    def _id() -> str:
        return str(uuid.uuid4())

    def capabilities(self, workload: Workload) -> Dict[str, Any]:
        return self._run("capabilities", request(self._id(), workload))

    def baseline(
        self,
        workload: Workload,
        measurement: MeasurementProfile,
        fingerprint: str,
    ) -> Dict[str, Any]:
        return self._run(
            "baseline",
            request(
                self._id(),
                workload,
                measurement=measurement,
                expected_fingerprint=fingerprint,
            ),
        )

    def evaluate(
        self,
        workload: Workload,
        plan: Dict[str, Any],
        measurement: MeasurementProfile,
        fingerprint: str,
    ) -> Dict[str, Any]:
        return self._run(
            "evaluate",
            request(
                self._id(),
                workload,
                plan=plan,
                measurement=measurement,
                expected_fingerprint=fingerprint,
            ),
        )

    def inspect(
        self, workload: Workload, plan: Dict[str, Any], fingerprint: str
    ) -> Dict[str, Any]:
        return self._run(
            "inspect",
            request(
                self._id(), workload, plan=plan, expected_fingerprint=fingerprint
            ),
        )

    def capture_aot(
        self,
        workload: Workload,
        plan: Dict[str, Any],
        fingerprint: str,
        bundle_dir: str | os.PathLike[str],
    ) -> Dict[str, Any]:
        """Ask a native runner to capture and link an AOT bundle.

        This deliberately has no compatibility fallback.  An older runner must
        fail export instead of silently producing a package that JITs at load or
        construction time.
        """
        result = self._run(
            "capture-aot",
            request(
                self._id(), workload, plan=plan, expected_fingerprint=fingerprint
            ),
            "--bundle-dir",
            str(Path(bundle_dir).resolve()),
        )
        bundle = result.get("aot_bundle")
        if result.get("status") == "captured" and isinstance(bundle, dict):
            images = bundle.get("images")
            if isinstance(images, list):
                root = Path(bundle_dir).resolve()
                for image in images:
                    if not isinstance(image, dict) or not isinstance(
                        image.get("file"), str
                    ):
                        continue
                    path = (root / image["file"]).resolve()
                    try:
                        path.relative_to(root)
                    except ValueError:
                        continue
                    if path.is_file():
                        image["size"] = path.stat().st_size
                        image["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def validate_aot(
        self,
        workload: Workload,
        fingerprint: str,
        package_dir: str | os.PathLike[str],
    ) -> Dict[str, Any]:
        return self._run(
            "validate-aot",
            request(self._id(), workload, expected_fingerprint=fingerprint),
            "--package",
            str(Path(package_dir).resolve()),
        )

    def perf_diagnose_stage(
        self,
        workload: Workload,
        plan: Dict[str, Any],
        fingerprint: str,
        stage: str,
        events: Mapping[str, str],
    ) -> Dict[str, Any]:
        """Run each semantic PMU role in a non-multiplexed controlled pass."""
        from .perf_diag import parse_perf_csv

        payload = request(
            self._id(), workload, plan=plan, expected_fingerprint=fingerprint
        )
        aggregate: Dict[str, Any] = {}
        representative: Dict[str, Any] | None = None
        timings: list[float] = []
        for role, event in events.items():
            request_id = str(payload["request_id"])
            path = ""
            control_read = control_write = ack_read = ack_write = -1
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", suffix=".json", delete=False
                ) as stream:
                    path = stream.name
                    stream.write(canonical_json(payload))
                    stream.write("\n")
                control_read, control_write = os.pipe()
                ack_read, ack_write = os.pipe()
                environment = os.environ.copy()
                environment.update(self.env)
                command = [
                    "perf", "stat", "-x", ";", "--no-big-num", "--delay=-1",
                    "--control", f"fd:{control_read},{ack_write}",
                    "-e", event, "--", self.executable, "perf-diag",
                    "--request", path, "--stage", stage,
                    "--perf-control-fd", str(control_write),
                    "--perf-ack-fd", str(ack_read),
                ]
                completed = subprocess.run(
                    command, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, timeout=self.timeout,
                    env=environment,
                    pass_fds=(control_read, control_write, ack_read, ack_write),
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return {"protocol_version": 1, "request_id": request_id,
                        "status": "timed_out", "detail": str(exc)}
            finally:
                for fd in (control_read, control_write, ack_read, ack_write):
                    if fd >= 0:
                        try:
                            os.close(fd)
                        except OSError:
                            pass
                if path:
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        pass
            if completed.returncode != 0:
                return {"protocol_version": 1, "request_id": request_id,
                        "status": "runner_error",
                        "exit_code": completed.returncode,
                        "stderr": completed.stderr}
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            if len(lines) != 1:
                return {"protocol_version": 1, "request_id": request_id,
                        "status": "protocol_error",
                        "detail": "runner stdout must contain one JSON object",
                        "stdout": completed.stdout, "stderr": completed.stderr}
            try:
                response = validate_response(json.loads(lines[0]), request_id)
            except (json.JSONDecodeError, ValueError) as exc:
                return {"protocol_version": 1, "request_id": request_id,
                        "status": "protocol_error", "detail": str(exc)}
            if representative is None:
                representative = response
            samples = response.get("samples_ms")
            if isinstance(samples, list):
                timings.append(float(sorted(samples)[len(samples) // 2]))
            parsed = parse_perf_csv(completed.stderr, {event: role})
            aggregate.update(parsed)
        if representative is None:
            return {"protocol_version": 1, "request_id": str(payload["request_id"]),
                    "status": "protocol_error", "detail": "empty PMU event set"}
        representative["pmu"] = aggregate
        if timings:
            representative["counter_pass_medians_ms"] = timings
            drift = (max(timings) - min(timings)) / max(
                sorted(timings)[len(timings) // 2], 1e-12)
            representative["counter_pass_timing_drift"] = drift
        return representative
