from __future__ import annotations

from typing import Any, Dict, Iterable, List


class RandomSearch:
    """Minimal ask/tell strategy over an already randomized finite domain."""

    def __init__(self, plans: Iterable[Dict[str, Any]]) -> None:
        self._plans: List[Dict[str, Any]] = list(plans)
        self._next = 0
        self.observations: List[tuple[Dict[str, Any], Dict[str, Any]]] = []

    def ask(self) -> Dict[str, Any]:
        if self._next >= len(self._plans):
            raise StopIteration
        plan = self._plans[self._next]
        self._next += 1
        return plan

    def tell(self, plan: Dict[str, Any], result: Dict[str, Any]) -> None:
        self.observations.append((plan, result))

