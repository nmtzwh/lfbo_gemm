from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Sequence

from .protocol import PLAN_SCHEMA_VERSION, Workload, stable_hash


def _unique(values: Iterable[int]) -> List[int]:
    result: List[int] = []
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
    ) -> None:
        self.workload = workload
        self.baseline = dict(baseline)
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
        for field, values in self.domains.items():
            if not values:
                raise ValueError(f"empty plan domain: {field}")

    @property
    def fields(self) -> Sequence[str]:
        return self.ordered_fields + self.categorical_fields

    def canonicalize(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(self.baseline)
        result.update(plan)
        result["schema_version"] = PLAN_SCHEMA_VERSION
        if result["pack_b"].endswith("n32"):
            result["N_blk"] = 32
        elif result["pack_b"].endswith("n64"):
            result["N_blk"] = 64
        return result

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
            if digest not in seen:
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
            if digest not in seen:
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
                index = domain.index(value) if value in domain else 0
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
) -> List[Dict[str, Any]]:
    if budget <= 0:
        return []
    space = PlanSpace(workload, baseline, capabilities)
    choices = space.domains
    proposed: List[Dict[str, Any]] = []
    seen = {stable_hash(baseline)}
    # Preserve all legal-looking one-field mutations before random combinations.
    for field, values in choices.items():
        for value in values:
            plan = space.canonicalize({field: value})
            digest = stable_hash(plan)
            if digest not in seen:
                seen.add(digest)
                proposed.append(plan)
    rng = random.Random(seed)
    attempts = 0
    while len(proposed) < budget and attempts < budget * 100:
        attempts += 1
        plan = space.random_plan(rng)
        digest = stable_hash(plan)
        if digest not in seen:
            seen.add(digest)
            proposed.append(plan)
    return proposed[:budget]
