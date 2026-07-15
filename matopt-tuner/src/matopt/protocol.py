from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict

PROTOCOL_VERSION = 1
PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Workload:
    m: int
    n: int
    k: int
    threads: int
    cpus: str
    dtype: str = "f32"
    layout: str = "dense_row_major"
    alpha: float = 1.0
    beta: float = 0.0

    def validate(self) -> None:
        if min(self.m, self.n, self.k, self.threads) <= 0:
            raise ValueError("dimensions and threads must be positive")
        if not self.cpus:
            raise ValueError("an explicit CPU mask is required")
        if (self.dtype, self.layout, self.alpha, self.beta) != (
            "f32",
            "dense_row_major",
            1.0,
            0.0,
        ):
            raise ValueError("only dense row-major FP32 alpha=1 beta=0 is supported")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class MeasurementProfile:
    profile: str = "macro"
    warmups: int = 3
    samples: int = 3
    minimum_sample_ms: float = 100.0
    seed: int = 19260817

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def request(
    request_id: str,
    workload: Workload,
    *,
    plan: Dict[str, Any] | None = None,
    measurement: MeasurementProfile | None = None,
    expected_fingerprint: str = "",
) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "expected_fingerprint": expected_fingerprint,
        "workload": workload.to_dict(),
    }
    if plan is not None:
        value["plan"] = plan
    if measurement is not None:
        value["measurement"] = measurement.to_dict()
    return value


def validate_response(value: Any, request_id: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("runner response is not an object")
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("runner protocol version mismatch")
    if value.get("request_id") != request_id:
        raise ValueError("runner request_id mismatch")
    if not isinstance(value.get("status"), str):
        raise ValueError("runner response has no status")
    return value

