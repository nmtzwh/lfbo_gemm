from __future__ import annotations

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

    def _run(self, command: str, payload: Dict[str, Any]) -> Dict[str, Any]:
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
                [self.executable, command, "--request", path],
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
