from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Sequence

from .protocol import PLAN_SCHEMA_VERSION, Workload, stable_hash
from .space_config import SpaceConfig


def _unique(values: Iterable[Any]) -> List[Any]:
    result: List[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


class PlanSpace:
    """Capability-derived mixed discrete MatOpt plan space."""

    ordered_fields = (
        "M_blk",
        "N_blk",
        "K_blk",
        "M_chunk_size",
        "N_chunk_size",
        "brgemm_batch_size",
        "nthr_k",
        "bd_block",
        "ld_block2",
    )
    categorical_fields = ("pack_a", "pack_b", "loop_order")

    def __init__(
        self,
        workload: Workload,
        baseline: Dict[str, Any],
        capabilities: Dict[str, Any],
        config: SpaceConfig | None = None,
    ) -> None:
        self.workload = workload
        self.baseline = dict(baseline)
        self.config = config
        if config is not None:
            config.validate_capabilities(capabilities)
        allow_split = capabilities["constraints"]["allow_split_k"]
        self.domains: Dict[str, List[Any]] = {
            "M_blk": [
                x
                for x in sorted(
                    {
                        baseline["M_blk"],
                        64,
                        96,
                        128,
                        160,
                        192,
                        224,
                        256,
                    }
                )
                if x <= workload.m
            ],
            "N_blk": [
                x
                for x in sorted({baseline["N_blk"], 32, 64})
                if x <= workload.n
            ],
            "K_blk": [
                x
                for x in sorted({baseline["K_blk"], 256, 512, 1024})
                if x <= workload.k
            ],
            "M_chunk_size": [baseline["M_chunk_size"]],
            "N_chunk_size": sorted({baseline["N_chunk_size"], 1, 2, 4, 8}),
            "brgemm_batch_size": sorted(
                {baseline["brgemm_batch_size"], 1, 2, 4}
            ),
            "nthr_k": (
                sorted({baseline["nthr_k"], 1, 2, 4}) if allow_split else [1]
            ),
            "pack_a": list(capabilities["domains"]["pack_a"]),
            "pack_b": list(capabilities["domains"]["pack_b"]),
            "bd_block": list(capabilities["domains"]["bd_block"]),
            "ld_block2": list(capabilities["domains"]["ld_block2"]),
            "loop_order": list(capabilities["domains"]["loop_order"]),
        }
        if config is not None:
            self._apply_config(config)
        self.conditions = tuple(config.conditions) if config is not None else ()
        self.scratchpad_limit = int(
            (config.limits if config is not None else {}).get(
                "scratchpad_per_thread_bytes",
                capabilities["constraints"].get(
                    "scratchpad_per_thread_bytes", 64 * 1024 * 1024
                ),
            )
        )
        self.minimum_parallel_work = float(
            (config.limits if config is not None else {}).get(
                "minimum_parallel_work_per_thread", 0.0
            )
        )
        for field, values in self.domains.items():
            if not values:
                raise ValueError(f"empty plan domain: {field}")
        self._validate_conditions()

    def _apply_config(self, config: SpaceConfig) -> None:
        dimension_fields = {
            "M_blk": self.workload.m,
            "N_blk": self.workload.n,
            "K_blk": self.workload.k,
        }
        for field, domain in config.domains.items():
            values = list(domain.values)
            if config.inherit_baseline:
                values.append(self.baseline[field])
            if field in self.ordered_fields:
                self.domains[field] = sorted(set(values))
            else:
                self.domains[field] = _unique(values)
        for field, dimension in dimension_fields.items():
            self.domains[field] = [
                value for value in self.domains[field] if value <= dimension
            ]
        self.domains["nthr_k"] = [
            value
            for value in self.domains["nthr_k"]
            if self.workload.threads % value == 0
        ]
        if self.domains["K_blk"]:
            max_k_blocks = max(
                (self.workload.k + k_blk - 1) // k_blk
                for k_blk in self.domains["K_blk"]
            )
            self.domains["brgemm_batch_size"] = [
                value
                for value in self.domains["brgemm_batch_size"]
                if value <= max_k_blocks
            ]

    def _validate_conditions(self) -> None:
        for condition in self.conditions:
            for field, values in condition.when.items():
                unsupported = set(values) - set(self.domains[field])
                if unsupported:
                    raise ValueError(
                        f"condition references values outside domain {field}: "
                        f"{sorted(unsupported)}"
                    )
            for field, value in condition.force.items():
                if value not in self.domains[field]:
                    raise ValueError(
                        f"condition forces value outside domain {field}: {value}"
                    )

    @property
    def fields(self) -> Sequence[str]:
        return self.ordered_fields + self.categorical_fields

    def canonicalize(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(self.baseline)
        result.update(plan)
        result["schema_version"] = PLAN_SCHEMA_VERSION
        self._apply_conditions(result)
        if result["pack_b"].endswith("n32"):
            result["N_blk"] = 32
        elif result["pack_b"].endswith("n64"):
            result["N_blk"] = 64
        return result

    def _apply_conditions(self, plan: Dict[str, Any]) -> None:
        for _ in range(len(self.conditions) + 1):
            changed = False
            for condition in self.conditions:
                if all(plan.get(field) in values for field, values in condition.when.items()):
                    for field, value in condition.force.items():
                        if plan.get(field) != value:
                            plan[field] = value
                            changed = True
            if not changed:
                return
        raise ValueError("SpaceConfig conditions do not converge")

    def _estimated_scratchpad(self, plan: Dict[str, Any]) -> int:
        size = 0
        if plan["pack_a"] == "per_call_padded":
            size += plan["M_blk"] * plan["K_blk"] * 4
        if plan["pack_b"] in {"per_call_n32", "per_call_n64"}:
            size += (
                plan["K_blk"]
                * plan["N_blk"]
                * plan["brgemm_batch_size"]
                * 4
            )
        if plan["nthr_k"] > 1:
            size += plan["M_blk"] * plan["N_blk"] * 4
        return size

    def is_allowed(self, plan: Dict[str, Any]) -> bool:
        if any(plan[field] not in self.domains[field] for field in self.fields):
            return False
        k_blocks = (self.workload.k + plan["K_blk"] - 1) // plan["K_blk"]
        if plan["brgemm_batch_size"] > k_blocks:
            return False
        k_chunks = (
            k_blocks + plan["brgemm_batch_size"] - 1
        ) // plan["brgemm_batch_size"]
        if plan["nthr_k"] > k_chunks:
            return False
        if self._estimated_scratchpad(plan) > self.scratchpad_limit:
            return False
        if self.minimum_parallel_work:
            m_units = (
                self.workload.m
                + plan["M_blk"] * plan["M_chunk_size"]
                - 1
            ) // (plan["M_blk"] * plan["M_chunk_size"])
            n_units = (
                self.workload.n
                + plan["N_blk"] * plan["N_chunk_size"]
                - 1
            ) // (plan["N_blk"] * plan["N_chunk_size"])
            work_per_thread = (
                m_units * n_units * min(plan["nthr_k"], k_chunks)
            ) / self.workload.threads
            if work_per_thread < self.minimum_parallel_work:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "space_schema_version": 1,
            "requested": self.config.to_dict() if self.config is not None else None,
            "effective_domains": {
                field: list(self.domains[field]) for field in self.fields
            },
            "limits": {
                "scratchpad_per_thread_bytes": self.scratchpad_limit,
                "minimum_parallel_work_per_thread": self.minimum_parallel_work,
            },
        }

    def random_plan(self, rng: random.Random) -> Dict[str, Any]:
        return self.canonicalize(
            {field: rng.choice(self.domains[field]) for field in self.fields}
        )

    def random_plans(
        self, count: int, rng: random.Random, excluded: set[str] | None = None
    ) -> List[Dict[str, Any]]:
        seen = set(excluded or ())
        result: List[Dict[str, Any]] = []
        attempts = 0
        while len(result) < count and attempts < max(100, count * 100):
            attempts += 1
            plan = self.random_plan(rng)
            digest = stable_hash(plan)
            if digest not in seen and self.is_allowed(plan):
                seen.add(digest)
                result.append(plan)
        return result

    def neighbors(
        self,
        base: Dict[str, Any],
        count: int,
        radius: int,
        rng: random.Random,
        excluded: set[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Random multi-field perturbations around a search copy.

        Ordered fields move by at most ``radius`` domain indices. Categorical
        fields may switch to any other category. Between one and ``radius``
        fields are changed in each proposal.
        """
        seen = set(excluded or ())
        seen.add(stable_hash(base))
        result: List[Dict[str, Any]] = []
        attempts = 0
        mutable = [field for field in self.fields if len(self.domains[field]) > 1]
        while len(result) < count and attempts < max(100, count * 100):
            attempts += 1
            plan = dict(base)
            changed = rng.sample(
                mutable, rng.randint(1, min(radius, len(mutable)))
            )
            for field in changed:
                domain = self.domains[field]
                current = plan[field]
                if field in self.ordered_fields and current in domain:
                    index = domain.index(current)
                    choices = [
                        value
                        for offset, value in enumerate(domain)
                        if value != current and abs(offset - index) <= radius
                    ]
                else:
                    choices = [value for value in domain if value != current]
                if choices:
                    plan[field] = rng.choice(choices)
            plan = self.canonicalize(plan)
            digest = stable_hash(plan)
            if digest not in seen and self.is_allowed(plan):
                seen.add(digest)
                result.append(plan)
        return result

    def encode(self, plans: Sequence[Dict[str, Any]]):
        """Encode ordered indices and one-hot categorical values for sklearn."""
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - exercised without extra
            raise RuntimeError(
                "LFBO requires numpy; install matopt-tuner[lfbo]"
            ) from exc
        rows: List[List[float]] = []
        for plan in plans:
            row: List[float] = []
            for field in self.ordered_fields:
                domain = self.domains[field]
                value = plan[field]
                if value in domain:
                    index = domain.index(value)
                else:
                    index = min(
                        range(len(domain)), key=lambda item: abs(domain[item] - value)
                    )
                row.append(index / max(1, len(domain) - 1))
            for field in self.categorical_fields:
                value = plan[field]
                row.extend(float(value == candidate) for candidate in self.domains[field])
            rows.append(row)
        return np.asarray(rows, dtype=np.float64)


def candidates(
    workload: Workload,
    baseline: Dict[str, Any],
    capabilities: Dict[str, Any],
    budget: int,
    seed: int,
    config: SpaceConfig | None = None,
) -> List[Dict[str, Any]]:
    if budget <= 0:
        return []
    space = PlanSpace(workload, baseline, capabilities, config)
    choices = space.domains
    proposed: List[Dict[str, Any]] = []
    seen = {stable_hash(baseline)}
    # Preserve all legal-looking one-field mutations before random combinations.
    for field, values in choices.items():
        for value in values:
            plan = space.canonicalize({field: value})
            digest = stable_hash(plan)
            if digest not in seen and space.is_allowed(plan):
                seen.add(digest)
                proposed.append(plan)
    rng = random.Random(seed)
    attempts = 0
    while len(proposed) < budget and attempts < budget * 100:
        attempts += 1
        plan = space.random_plan(rng)
        digest = stable_hash(plan)
        if digest not in seen and space.is_allowed(plan):
            seen.add(digest)
            proposed.append(plan)
    return proposed[:budget]
