from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Protocol, Sequence

from ..protocol import stable_hash
from ..space import PlanSpace


class AskTellOptimizer(Protocol):
    def ask(self) -> Dict[str, Any]: ...
    def tell(self, plan: Dict[str, Any], result: Dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class LFBOConfig:
    initial_population: int = 24
    copies: int = 3
    generations: int = 8
    neighbors: int = 256
    radius: int = 2
    quantile: float = 0.10
    fraction_selected: float = 0.10
    similarity_penalty: float = 1.0
    patience: int = 2
    trees: int = 100
    model_jobs: int = 1

    def validate(self) -> None:
        if min(
            self.initial_population,
            self.copies,
            self.generations,
            self.neighbors,
            self.radius,
            self.patience,
            self.trees,
            self.model_jobs,
        ) <= 0:
            raise ValueError("LFBO integer parameters must be positive")
        if not 0 < self.quantile < 1:
            raise ValueError("LFBO quantile must be in (0, 1)")
        if not 0 < self.fraction_selected <= 1:
            raise ValueError("LFBO fraction_selected must be in (0, 1]")
        if self.similarity_penalty < 0:
            raise ValueError("LFBO similarity_penalty must be non-negative")


@dataclass
class Observation:
    plan: Dict[str, Any]
    loss: float
    status: str


class LFBOSearch:
    """Helion-inspired LFBO pattern search for MatOpt plans.

    A RandomForest classifier learns whether a plan belongs to the best
    ``quantile`` of finite observations. Failed, rejected, incorrect, crashed,
    and timed-out plans are retained as negative examples with infinite loss.
    Candidate batches are selected greedily by predicted probability minus
    RandomForest leaf-cooccurrence similarity to earlier selections.
    """

    def __init__(
        self,
        space: PlanSpace,
        *,
        objective: str,
        budget: int,
        seed: int = 19260817,
        config: LFBOConfig | None = None,
    ) -> None:
        try:
            import numpy  # noqa: F401
            import sklearn  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "LFBO requires numpy and scikit-learn; install "
                "matopt-tuner[lfbo] or run `uv sync --extra lfbo`"
            ) from exc
        if objective not in {"one_shot", "steady", "throughput"}:
            raise ValueError("invalid LFBO objective")
        if budget <= 0:
            raise ValueError("LFBO budget must be positive")
        self.space = space
        self.objective = objective
        self.budget = budget
        self.random_seed = seed
        self.config = config or LFBOConfig()
        self.config.validate()
        self.rng = random.Random(seed)
        self.observations: List[Observation] = []
        self.visited: set[str] = set()
        self.queue: List[Dict[str, Any]] = []
        self.model = None
        self.generation = 0
        self.asked = 0
        self.best_loss = math.inf
        self.stale_generations = 0
        self.last_generation_best = math.inf
        self.proposals: Dict[str, Dict[str, Any]] = {}

    def _loss(self, result: Dict[str, Any]) -> float:
        if result.get("status") != "benchmarked":
            return math.inf
        measurement = result.get("measurement", {})
        if not measurement.get("correct", False):
            return math.inf
        if self.objective == "one_shot":
            return float(measurement["one_shot_ms"])
        if self.objective == "steady":
            return float(measurement["steady_ms"])
        throughput = float(measurement["throughput_gflops"])
        return 1.0 / throughput if throughput > 0 else math.inf

    def seed(self, plan: Dict[str, Any], result: Dict[str, Any]) -> None:
        digest = stable_hash(plan)
        if digest in self.visited:
            return
        loss = self._loss(result)
        self.visited.add(digest)
        self.observations.append(
            Observation(dict(plan), loss, str(result.get("status", "unknown")))
        )
        if loss < self.best_loss:
            self.best_loss = loss

    def tell(self, plan: Dict[str, Any], result: Dict[str, Any]) -> None:
        self.seed(plan, result)

    def restore_budget_used(self, evaluations: int) -> None:
        """Restore the number of non-baseline evaluations from JSONL."""
        self.asked = min(max(0, evaluations), self.budget)

    def proposal_metadata(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        return dict(self.proposals.get(stable_hash(plan), {}))

    def summary(self) -> Dict[str, Any]:
        return {
            "algorithm": "lfbo_pattern_search",
            "config": asdict(self.config),
            "budget": self.budget,
            "budget_used": self.asked,
            "generations": self.generation,
            "observations": len(self.observations),
            "failed_observations": sum(
                not math.isfinite(item.loss) for item in self.observations
            ),
            "surrogate_fitted": self.model is not None,
        }

    def _fit(self) -> None:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        if len(self.observations) < 2:
            self.model = None
            return
        losses = np.asarray([item.loss for item in self.observations], dtype=float)
        finite = np.isfinite(losses)
        if not finite.any():
            self.model = None
            return
        threshold = float(np.quantile(losses[finite], self.config.quantile))
        positive = finite & (losses <= threshold)
        labels = positive.astype(float)
        if np.all(labels == labels[0]):
            self.model = None
            return
        positive_weights = np.maximum(1e-5, threshold - losses)
        positive_weights[~positive] = 0
        mean_positive = float(np.mean(positive_weights[positive]))
        positive_weights /= max(mean_positive, 1e-5)
        sample_weight = np.where(positive, positive_weights, 1.0)
        model = RandomForestClassifier(
            criterion="log_loss",
            random_state=self.random_seed + self.generation,
            n_estimators=self.config.trees,
            n_jobs=self.config.model_jobs,
        )
        model.fit(
            self.space.encode([item.plan for item in self.observations]),
            labels,
            sample_weight=sample_weight,
        )
        self.model = model

    def _select_diverse(
        self, plans: Sequence[Dict[str, Any]], count: int
    ) -> List[Dict[str, Any]]:
        import numpy as np

        if not plans or count <= 0:
            return []
        count = min(count, len(plans))
        model = self.model
        if model is None:
            indices = list(range(len(plans)))
            self.rng.shuffle(indices)
            return [plans[index] for index in indices[:count]]
        encoded = self.space.encode(plans)
        probability = np.asarray(model.predict_proba(encoded))[:, 1]
        leaves = model.apply(encoded)
        similarity_sums = np.zeros(len(plans), dtype=float)
        remaining = list(range(len(plans)))
        selected: List[int] = []
        while remaining and len(selected) < count:
            if selected:
                similarity = similarity_sums[remaining] / len(selected)
                score = probability[remaining] - self.config.similarity_penalty * similarity
            else:
                score = probability[remaining]
            local = int(np.argmax(score))
            chosen = remaining.pop(local)
            selected.append(chosen)
            same_leaf = leaves == leaves[chosen : chosen + 1]
            similarity_sums += same_leaf.sum(axis=1) / leaves.shape[1]
        return [plans[index] for index in selected]

    def _initial_batch(self) -> List[Dict[str, Any]]:
        remaining = min(self.config.initial_population, self.budget - self.asked)
        return self.space.random_plans(remaining, self.rng, self.visited)

    def _generation_batch(self) -> List[Dict[str, Any]]:
        if self.generation >= self.config.generations:
            return []
        finite = [
            item
            for item in self.observations
            if math.isfinite(item.loss) and self.space.is_allowed(item.plan)
        ]
        if not finite:
            return self._initial_batch()
        finite.sort(key=lambda item: item.loss)
        copies = finite[: self.config.copies]
        pool: List[Dict[str, Any]] = []
        pool_hashes = set(self.visited)
        per_copy = max(1, math.ceil(self.config.neighbors / len(copies)))
        for item in copies:
            generated = self.space.neighbors(
                item.plan,
                per_copy,
                self.config.radius,
                self.rng,
                pool_hashes,
            )
            pool.extend(generated)
            pool_hashes.update(stable_hash(plan) for plan in generated)
        if not pool:
            return []
        self._fit()
        selected_count = max(1, math.ceil(len(pool) * self.config.fraction_selected))
        selected_count = min(selected_count, self.budget - self.asked)
        self.generation += 1
        return self._select_diverse(pool, selected_count)

    def ask(self) -> Dict[str, Any]:
        if self.asked >= self.budget:
            raise StopIteration
        if not self.queue:
            if self.asked == 0:
                self.last_generation_best = self.best_loss
                self.queue = self._initial_batch()
                for queued in self.queue:
                    self.proposals[stable_hash(queued)] = {
                        "algorithm": "lfbo_pattern_search",
                        "phase": "initial_population",
                        "generation": 0,
                    }
            else:
                relative = (
                    abs(self.last_generation_best / self.best_loss - 1.0)
                    if math.isfinite(self.last_generation_best)
                    and self.best_loss > 0
                    else math.inf
                )
                self.stale_generations = (
                    self.stale_generations + 1 if relative < 0.001 else 0
                )
                if self.stale_generations >= self.config.patience:
                    raise StopIteration
                self.last_generation_best = self.best_loss
                self.queue = self._generation_batch()
                for queued in self.queue:
                    self.proposals[stable_hash(queued)] = {
                        "algorithm": "lfbo_pattern_search",
                        "phase": "surrogate_selected",
                        "generation": self.generation,
                    }
            if not self.queue:
                raise StopIteration
        plan = self.queue.pop(0)
        self.asked += 1
        return plan


class ExternalLFBOSearch:
    """Compatibility adapter for a third-party ask/tell optimizer."""

    def __init__(self, optimizer: AskTellOptimizer) -> None:
        self.optimizer = optimizer

    def ask(self) -> Dict[str, Any]:
        return self.optimizer.ask()

    def tell(self, plan: Dict[str, Any], result: Dict[str, Any]) -> None:
        self.optimizer.tell(plan, result)
