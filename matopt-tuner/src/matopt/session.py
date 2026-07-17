from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

from .history import History
from .objectives import pareto, select
from .protocol import MeasurementProfile, Workload, canonical_json, stable_hash
from .runner import MatOptRunner
from .search.lfbo import LFBOConfig, LFBOSearch
from .search.random import RandomSearch
from .space import PlanSpace, candidates
from .space_config import SpaceConfig


class TuningSession:
    def __init__(
        self,
        workload: Workload,
        runner: MatOptRunner,
        *,
        history: str | os.PathLike[str],
    ) -> None:
        self.workload = workload
        self.runner = runner
        self.history_path = Path(history)

    @staticmethod
    def _record(
        fingerprint: str,
        state: str,
        plan: Dict[str, Any],
        response: Dict[str, Any] | None = None,
        phase: str = "search",
        search_metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "fingerprint": fingerprint,
            "timestamp_ns": time.time_ns(),
            "state": state,
            "phase": phase,
            "plan_hash": stable_hash(plan),
            "plan": plan,
        }
        if response is not None:
            value["response"] = response
        if search_metadata:
            value["search"] = search_metadata
        return value

    def tune(
        self,
        *,
        budget: int = 256,
        objective: str = "one_shot",
        seed: int = 19260817,
        measurement: MeasurementProfile | None = None,
        search: Any | None = None,
        algorithm: str = "random",
        lfbo_config: LFBOConfig | None = None,
        space_config: SpaceConfig | None = None,
        progress: Callable[[str, Dict[str, Any]], None] | None = None,
        output: str | os.PathLike[str] | None = None,
    ) -> Dict[str, Any]:
        if objective not in {"one_shot", "steady", "throughput"}:
            raise ValueError("invalid objective")
        measurement = measurement or MeasurementProfile()
        caps = self.runner.capabilities(self.workload)
        if caps.get("status") != "capabilities":
            raise RuntimeError(f"capability discovery failed: {caps}")
        runner_fingerprint = str(caps["fingerprint"])

        baseline_response = self.runner.baseline(
            self.workload, measurement, runner_fingerprint
        )
        if baseline_response.get("status") != "benchmarked":
            raise RuntimeError(f"baseline failed: {baseline_response}")
        baseline_plan = baseline_response["effective_plan"]
        space = PlanSpace(self.workload, baseline_plan, caps, space_config)
        space_description = space.to_dict()
        space_hash = stable_hash(
            {
                "space": space_description,
                "runner_space_capabilities": {
                    "constraints": caps.get("constraints", {}),
                    "domains": caps.get("domains", {}),
                },
            }
        )
        fingerprint = (
            stable_hash(
                {
                    "runner_fingerprint": runner_fingerprint,
                    "space_hash": space_hash,
                }
            )
            if space_config is not None
            else runner_fingerprint
        )
        history = History(self.history_path, fingerprint)
        records = history.load()
        completed = history.completed(records)
        baseline_hash = stable_hash(baseline_plan)
        if baseline_hash in completed:
            baseline_record = completed[baseline_hash]
        else:
            history.append(
                self._record(fingerprint, "started", baseline_plan, phase="baseline")
            )
            baseline_record = self._record(
                fingerprint,
                "benchmarked",
                baseline_plan,
                baseline_response,
                phase="baseline",
            )
            history.append(baseline_record)
            records.append(baseline_record)
        completed[baseline_hash] = baseline_record

        if progress is not None:
            progress(
                "start",
                {
                    "workload": self.workload,
                    "capabilities": caps,
                    "space": space_description,
                    "objective": objective,
                    "algorithm": (
                        algorithm if search is None else type(search).__name__
                    ),
                    "budget": budget,
                    "resumed": sum(
                        plan_hash != baseline_hash for plan_hash in completed
                    ),
                },
            )
            progress(
                "baseline",
                {
                    "record": baseline_record,
                    "prior_records": [
                        record
                        for plan_hash, record in completed.items()
                        if plan_hash != baseline_hash
                    ],
                },
            )

        if search is None:
            if algorithm == "random":
                search = RandomSearch(
                    candidates(
                        self.workload,
                        baseline_plan,
                        caps,
                        budget,
                        seed,
                        space_config,
                    )
                )
            elif algorithm == "lfbo":
                search = LFBOSearch(
                    space,
                    objective=objective,
                    budget=budget,
                    seed=seed,
                    config=lfbo_config,
                )
            else:
                raise ValueError(f"unknown search algorithm: {algorithm}")
        seed_search = getattr(search, "seed", None)
        if seed_search is not None:
            for record in sorted(
                completed.values(), key=lambda item: item.get("timestamp_ns", 0)
            ):
                seed_search(
                    record["plan"],
                    record.get("response", {"status": record["state"]}),
                )
        restore_budget = getattr(search, "restore_budget_used", None)
        if restore_budget is not None:
            restore_budget(sum(plan_hash != baseline_hash for plan_hash in completed))
        evaluated: List[Dict[str, Any]] = sorted(
            completed.values(), key=lambda item: item.get("timestamp_ns", 0)
        )
        for _ in range(budget):
            try:
                plan = search.ask()
            except StopIteration:
                break
            digest = stable_hash(plan)
            if digest in completed:
                record = completed[digest]
                evaluated.append(record)
                search.tell(plan, record.get("response", {"status": record["state"]}))
                continue
            metadata_fn = getattr(search, "proposal_metadata", None)
            search_metadata = metadata_fn(plan) if metadata_fn else None
            history.append(
                self._record(
                    fingerprint,
                    "started",
                    plan,
                    search_metadata=search_metadata,
                )
            )
            response = self.runner.evaluate(
                self.workload, plan, measurement, runner_fingerprint
            )
            state = str(response.get("status", "protocol_error"))
            record = self._record(
                fingerprint,
                state,
                plan,
                response,
                search_metadata=search_metadata,
            )
            history.append(record)
            records.append(record)
            completed[digest] = record
            evaluated.append(record)
            search.tell(plan, response)
            if progress is not None:
                progress("candidate", {"record": record})

        benchmarked = [r for r in evaluated if r.get("state") == "benchmarked"]
        selected = select(benchmarked, objective, baseline_record)
        frontier = pareto(benchmarked)
        result = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "runner_fingerprint": runner_fingerprint,
            "space_hash": space_hash,
            "space": space_description,
            "capabilities": caps,
            "workload": self.workload.to_dict(),
            "baseline": baseline_record,
            "selected": {objective: selected},
            "pareto": frontier,
            "search": (
                search.summary()
                if hasattr(search, "summary")
                else {"algorithm": type(search).__name__}
            ),
        }
        if output is not None:
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=destination.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(canonical_json(result))
                stream.write("\n")
            os.replace(temporary, destination)
        if progress is not None:
            progress("finish", {"result": result})
        return result
