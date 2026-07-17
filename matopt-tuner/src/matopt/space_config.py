from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .protocol import canonical_json, stable_hash


SPACE_SCHEMA_VERSION = 1

CONFIGURABLE_FIELDS = {
    "M_blk",
    "N_blk",
    "K_blk",
    "M_chunk_size",
    "N_chunk_size",
    "brgemm_batch_size",
    "nthr_k",
    "pack_a",
    "pack_b",
    "bd_block",
    "ld_block2",
    "loop_order",
}

INTEGER_FIELDS = CONFIGURABLE_FIELDS - {"pack_a", "pack_b", "loop_order"}
CAPABILITY_DOMAINS = {"pack_a", "pack_b", "bd_block", "ld_block2", "loop_order"}
CAPABILITY_FLAGS = {"allow_split_k"}
LIMIT_FIELDS = {
    "scratchpad_per_thread_bytes",
    "minimum_parallel_work_per_thread",
}


@dataclass(frozen=True)
class DomainConfig:
    values: tuple[Any, ...]
    require_capability: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {"values": list(self.values)}
        if self.require_capability is not None:
            value["require_capability"] = self.require_capability
        return value


@dataclass(frozen=True)
class ConditionConfig:
    when: Dict[str, tuple[Any, ...]]
    force: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "if": {key: list(values) for key, values in sorted(self.when.items())},
            "force": dict(sorted(self.force.items())),
        }


@dataclass(frozen=True)
class SpaceConfig:
    schema_version: int = SPACE_SCHEMA_VERSION
    inherit_baseline: bool = True
    domains: Dict[str, DomainConfig] = field(default_factory=dict)
    conditions: tuple[ConditionConfig, ...] = ()
    limits: Dict[str, float | int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "SpaceConfig":
        source = Path(path)
        text = source.read_text(encoding="utf-8")
        if source.suffix.lower() == ".json":
            value = json.loads(text)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    "YAML SpaceConfig requires PyYAML; install the project "
                    "dependencies or use a .json configuration"
                ) from exc
            value = yaml.safe_load(text)
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls, value: Any) -> "SpaceConfig":
        if not isinstance(value, Mapping):
            raise ValueError("SpaceConfig root must be an object")
        allowed = {
            "space_schema_version",
            "inherit_baseline",
            "domains",
            "conditions",
            "limits",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown SpaceConfig fields: {sorted(unknown)}")
        version = value.get("space_schema_version", SPACE_SCHEMA_VERSION)
        if version != SPACE_SCHEMA_VERSION:
            raise ValueError(f"unsupported SpaceConfig schema version: {version}")
        inherit = value.get("inherit_baseline", True)
        if not isinstance(inherit, bool):
            raise ValueError("inherit_baseline must be a boolean")

        raw_domains = value.get("domains", {})
        if not isinstance(raw_domains, Mapping):
            raise ValueError("SpaceConfig domains must be an object")
        domains: Dict[str, DomainConfig] = {}
        for name, raw in raw_domains.items():
            cls._check_field(name)
            if isinstance(raw, list):
                values = raw
                requirement = None
            elif isinstance(raw, Mapping):
                extra = set(raw) - {"values", "require_capability"}
                if extra:
                    raise ValueError(
                        f"unknown domain options for {name}: {sorted(extra)}"
                    )
                values = raw.get("values")
                requirement = raw.get("require_capability")
            else:
                raise ValueError(f"domain {name} must be a list or object")
            if not isinstance(values, list) or not values:
                raise ValueError(f"domain {name} must contain a non-empty values list")
            normalized = tuple(cls._normalize_value(name, item) for item in values)
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"domain {name} contains duplicate values")
            if requirement is not None:
                if not isinstance(requirement, str):
                    raise ValueError("require_capability must be a string")
                if requirement not in CAPABILITY_FLAGS:
                    raise ValueError(
                        f"unknown required capability for {name}: {requirement}"
                    )
            domains[name] = DomainConfig(normalized, requirement)

        raw_conditions = value.get("conditions", [])
        if not isinstance(raw_conditions, list):
            raise ValueError("SpaceConfig conditions must be a list")
        conditions: List[ConditionConfig] = []
        for index, raw in enumerate(raw_conditions):
            if not isinstance(raw, Mapping) or set(raw) != {"if", "force"}:
                raise ValueError(
                    f"condition {index} must contain exactly 'if' and 'force'"
                )
            when_raw, force_raw = raw["if"], raw["force"]
            if not isinstance(when_raw, Mapping) or not when_raw:
                raise ValueError(f"condition {index} if clause must be an object")
            if not isinstance(force_raw, Mapping) or not force_raw:
                raise ValueError(f"condition {index} force clause must be an object")
            when: Dict[str, tuple[Any, ...]] = {}
            for name, choices in when_raw.items():
                cls._check_field(name)
                if not isinstance(choices, list) or not choices:
                    raise ValueError(
                        f"condition {index} field {name} must be a non-empty list"
                    )
                when[name] = tuple(
                    cls._normalize_value(name, item) for item in choices
                )
            force: Dict[str, Any] = {}
            for name, item in force_raw.items():
                cls._check_field(name)
                force[name] = cls._normalize_value(name, item)
            conditions.append(ConditionConfig(when, force))

        raw_limits = value.get("limits", {})
        if not isinstance(raw_limits, Mapping):
            raise ValueError("SpaceConfig limits must be an object")
        unknown_limits = set(raw_limits) - LIMIT_FIELDS
        if unknown_limits:
            raise ValueError(f"unknown SpaceConfig limits: {sorted(unknown_limits)}")
        limits: Dict[str, float | int] = {}
        for name, item in raw_limits.items():
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"limit {name} must be numeric")
            if item <= 0:
                raise ValueError(f"limit {name} must be positive")
            limits[name] = int(item) if name.endswith("_bytes") else float(item)
        return cls(version, inherit, domains, tuple(conditions), limits)

    @staticmethod
    def _check_field(name: Any) -> None:
        if not isinstance(name, str) or name not in CONFIGURABLE_FIELDS:
            raise ValueError(f"unknown or unsupported search-space field: {name}")

    @staticmethod
    def _normalize_value(name: str, value: Any) -> Any:
        if name in INTEGER_FIELDS:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"domain value for {name} must be an integer")
            invalid = value < 0 if name in {"bd_block", "ld_block2"} else value <= 0
            if invalid:
                raise ValueError(f"domain value for {name} is out of range: {value}")
            return value
        if not isinstance(value, str) or not value:
            raise ValueError(f"domain value for {name} must be a string")
        return value

    def validate_capabilities(self, capabilities: Mapping[str, Any]) -> None:
        constraints = capabilities.get("constraints", {})
        capability_domains = capabilities.get("domains", {})
        for name, domain in self.domains.items():
            if domain.require_capability and not constraints.get(
                domain.require_capability, False
            ):
                raise ValueError(
                    f"domain {name} requires unavailable capability "
                    f"{domain.require_capability}"
                )
            if name in CAPABILITY_DOMAINS:
                allowed = set(capability_domains.get(name, []))
                unsupported = set(domain.values) - allowed
                if unsupported:
                    raise ValueError(
                        f"domain {name} contains runner-unsupported values: "
                        f"{sorted(unsupported)}"
                    )
        if "nthr_k" in self.domains and not constraints.get("allow_split_k", False):
            unsupported = set(self.domains["nthr_k"].values) - {1}
            if unsupported:
                raise ValueError("nthr_k>1 requires runner split-K capability")
        requested_scratch = self.limits.get("scratchpad_per_thread_bytes")
        runner_scratch = constraints.get("scratchpad_per_thread_bytes")
        if requested_scratch and runner_scratch and requested_scratch > runner_scratch:
            raise ValueError(
                "SpaceConfig scratchpad limit exceeds the runner capability"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "space_schema_version": self.schema_version,
            "inherit_baseline": self.inherit_baseline,
            "domains": {
                key: value.to_dict() for key, value in sorted(self.domains.items())
            },
            "conditions": [condition.to_dict() for condition in self.conditions],
            "limits": dict(sorted(self.limits.items())),
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def hash(self) -> str:
        return stable_hash(self.to_dict())
