from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .protocol import canonical_json


class History:
    def __init__(self, path: str | os.PathLike[str], fingerprint: str) -> None:
        self.path = Path(path)
        self.fingerprint = fingerprint

    def load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        lines = raw.splitlines(keepends=True)
        records: List[Dict[str, Any]] = []
        for index, line in enumerate(lines):
            complete = line.endswith(b"\n") or line.endswith(b"\r")
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                if index == len(lines) - 1 and not complete:
                    recovered = b"".join(lines[:index])
                    self.path.write_bytes(recovered)
                    break
                raise ValueError(f"malformed complete JSONL record {index + 1}")
            if record.get("fingerprint") != self.fingerprint:
                raise ValueError("history fingerprint mismatch")
            records.append(record)
        return records

    def append(self, record: Dict[str, Any]) -> None:
        if record.get("fingerprint") != self.fingerprint:
            raise ValueError("record fingerprint mismatch")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(record))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def completed(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for record in records:
            plan_hash = record.get("plan_hash")
            if plan_hash and record.get("state") in {
                "benchmarked",
                "incorrect",
                "rejected",
                "crashed",
                "timed_out",
                "protocol_error",
                "runner_error",
            }:
                result[str(plan_hash)] = record
        return result
