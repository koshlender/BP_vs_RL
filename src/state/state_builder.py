from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

from src.thesis.chapter_constants import CH4_QUEUE_CATEGORIES, CH4_QUEUE_ID_TO_LABEL


def chapter4_queue_category(queue: float):
    """Return the symbolic Chapter 4 queue category for q_i'(t).

    Thesis Chapter 4 lists six symbolic categories k,l,m,n,o,p. Its strict
    lower-bound notation creates gaps at 11,21,31,41 for integer queue lengths;
    this implementation uses the only consecutive integer partition consistent
    with the stated 10-vehicle buckets and upper limits:
    0..10, 11..20, 21..30, 31..40, 41..50, and >50.
    """
    q = float(queue)
    if q < 0 or int(q) != q:
        raise ValueError("Chapter 4 queue discretization expects non-negative integer queue lengths")
    q_int = int(q)
    for category in CH4_QUEUE_CATEGORIES:
        if q_int >= category.lower_inclusive and (
            category.upper_inclusive is None or q_int <= category.upper_inclusive
        ):
            return category
    raise ValueError(f"queue length {queue} was not assigned to any Chapter 4 category")


def chapter4_quantize_queue(queue: float) -> int:
    """Chapter 4 categorical state id for q_i(t), not a physical queue value."""
    return chapter4_queue_category(queue).category_id


def chapter4_queue_label(queue: float) -> str:
    """Chapter 4 symbolic queue label k,l,m,n,o,p for q_i(t)."""
    return chapter4_queue_category(queue).label


def chapter4_category_label(category_id: int) -> str:
    return CH4_QUEUE_ID_TO_LABEL[category_id]


def chapter4_local_state(queues: Iterable[float]) -> tuple[int, int, int, int]:
    """Thesis Chapter 4 independent/non-cooperative state: local incoming queues."""
    values = tuple(chapter4_quantize_queue(q) for q in queues)
    if len(values) != 4:
        raise ValueError("Chapter 4 local state must contain four incoming queues ordered N,S,E,W")
    return values


@dataclass(frozen=True)
class StateSpec:
    local_incoming: tuple[str, ...]
    neighbour_incoming: tuple[str, ...]
    neighbour_actions: tuple[str, ...]


class StateBuilder:
    """Builds Thesis Chapter 4 Eq. (4.1) cooperative state S(t)=[Q(t),Q'(t),a'(t)].

    Queue entries are Chapter 4 categorical state ids with an explicit mapping to
    symbolic labels k,l,m,n,o,p; they are not raw queue lengths. Ordering is
    exactly local queues Q(t), neighbouring queues Q'(t), then neighbouring
    non-cooperative green allocations a'(t).
    """

    def __init__(self, spec: StateSpec):
        self.spec = spec

    @property
    def dimension(self) -> int:
        return len(self.spec.local_incoming) + len(self.spec.neighbour_incoming) + len(self.spec.neighbour_actions)

    def build(self, queues: dict[str, float], neighbour_actions: dict[str, int | float]) -> list[float]:
        # Thesis Chapter 4, Eq. (4.1): S(t)=[Q(t),Q'(t),a'(t)].
        vals: list[float] = []
        for edge in self.spec.local_incoming:
            vals.append(chapter4_quantize_queue(queues.get(edge, 0.0)))
        for edge in self.spec.neighbour_incoming:
            vals.append(chapter4_quantize_queue(queues.get(edge, 0.0)))
        for connecting_road in self.spec.neighbour_actions:
            vals.append(float(neighbour_actions.get(connecting_road, 0.0)))
        return vals
