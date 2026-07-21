from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

from src.thesis.chapter_constants import CH4_QUEUE_CATEGORIES, CH4_QUEUE_ID_TO_LABEL


def chapter4_queue_category(queue: float):
    """Return the Chapter 4 symbolic queue category for q_i'(t).

    Chapter 4 quantizes each incoming-road queue as k,l,m,n,o,p using 10-vehicle
    buckets up to 50 and p for queues greater than 50. The thesis notation uses
    strict lower bounds that omit exact integer boundaries 11,21,31,41; this
    implementation assigns those integers to the next bucket to form a complete
    integer partition: 0..10, 11..20, 21..30, 31..40, 41..50, and >50.
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
    """Chapter 4 categorical state id for symbolic q_i(t), not a raw queue."""
    return chapter4_queue_category(queue).category_id


def chapter4_queue_label(queue: float) -> str:
    """Chapter 4 symbolic queue label k,l,m,n,o,p for q_i(t)."""
    return chapter4_queue_category(queue).label


def chapter4_category_label(category_id: int) -> str:
    return CH4_QUEUE_ID_TO_LABEL[category_id]


def chapter4_local_state(queues: Iterable[float]) -> tuple[int, int, int, int]:
    """Independent learner state: local incoming queues Q(t)=[q_N,q_S,q_E,q_W]."""
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
    """Build Chapter 4 cooperative state S(t)=[Q(t),Q'(t),a'(t)].

    Q(t) contains all local incoming queues, Q'(t) contains queues at neighboring
    non-cooperative intersections excluding connecting-road queues, and a'(t)
    contains the neighboring non-cooperative green allocations to connecting
    roads. Queue entries are Chapter 4 categorical ids for k,l,m,n,o,p.
    """

    def __init__(self, spec: StateSpec):
        self.spec = spec

    @property
    def dimension(self) -> int:
        return len(self.spec.local_incoming) + len(self.spec.neighbour_incoming) + len(self.spec.neighbour_actions)

    def build(self, queues: dict[str, float], neighbour_actions: dict[str, int | float]) -> list[float]:
        vals: list[float] = []
        for edge in self.spec.local_incoming:
            vals.append(chapter4_quantize_queue(queues.get(edge, 0.0)))
        for edge in self.spec.neighbour_incoming:
            vals.append(chapter4_quantize_queue(queues.get(edge, 0.0)))
        for connecting_road in self.spec.neighbour_actions:
            vals.append(float(neighbour_actions.get(connecting_road, 0.0)))
        return vals
