from __future__ import annotations

import itertools
from dataclasses import dataclass

from src.thesis.chapter_constants import (
    CH4_GREEN_TIMES_SECONDS,
    THESIS_CYCLE_TIME_SECONDS,
    THESIS_TOTAL_YELLOW_TIME_SECONDS,
)

# Thesis Chapter 4, Non Cooperative Signal / Cooperative Signals, Action.
# Green allocations are vectors [g_N,g_S,g_E,g_W] with each component in
# {4,8,12,16,20,24,28,32} and sum_j g_j(t)=T-Y. T and Y are imported from the
# shared thesis-cycle configuration instead of being redefined here.
APPROACH_ORDER = ("N", "S", "E", "W")

# Thesis Chapter 4, Cooperative Signals, Action: eight cyclic phase sequences,
# four possible starts and two directions (clockwise/counter-clockwise). The thesis
# gives the N->E->... clockwise example; the opposite direction is anti-clockwise.
CLOCKWISE_APPROACH_ORDER = ("N", "E", "S", "W")
COUNTERCLOCKWISE_APPROACH_ORDER = ("N", "W", "S", "E")


def _rotate(order: tuple[str, ...], start: str) -> tuple[str, ...]:
    idx = order.index(start)
    return order[idx:] + order[:idx]


def chapter4_phase_sequences() -> tuple[tuple[str, str, str, str], ...]:
    """Thesis Chapter 4, Cooperative Signals, Action: |P|=8 cyclic sequences."""
    sequences: list[tuple[str, str, str, str]] = []
    for start in CLOCKWISE_APPROACH_ORDER:
        sequences.append(_rotate(CLOCKWISE_APPROACH_ORDER, start))
        sequences.append(_rotate(COUNTERCLOCKWISE_APPROACH_ORDER, start))
    return tuple(sequences)


CH4_PHASE_SEQUENCES = chapter4_phase_sequences()


def chapter4_green_time_actions(
    cycle_time: int = THESIS_CYCLE_TIME_SECONDS,
    total_yellow_time: int = THESIS_TOTAL_YELLOW_TIME_SECONDS,
) -> tuple[tuple[int, int, int, int], ...]:
    """Thesis Chapter 4 green allocation action space.

    Ambiguity: Chapter 4 defines sum_j g_j(t)=T-Y but does not restate numeric
    T or Y. Defaults retain Chapter 3's T=80 and Y=16 unless callers pass values.
    """
    available_green = cycle_time - total_yellow_time
    return tuple(
        action
        for action in itertools.product(CH4_GREEN_TIMES_SECONDS, repeat=4)
        if sum(action) == available_green
    )


def chapter4_cooperative_actions(
    cycle_time: int = THESIS_CYCLE_TIME_SECONDS,
    total_yellow_time: int = THESIS_TOTAL_YELLOW_TIME_SECONDS,
) -> tuple[tuple[tuple[int, int, int, int], tuple[str, str, str, str]], ...]:
    """Thesis Chapter 4 cooperative action [G(t),p(t)] for p in P."""
    return tuple(
        (green_action, phase_sequence)
        for green_action in chapter4_green_time_actions(cycle_time, total_yellow_time)
        for phase_sequence in CH4_PHASE_SEQUENCES
    )


@dataclass(frozen=True)
class PhasePlan:
    green_phases: dict[int, int]
    yellow_phases: dict[tuple[int, int], int]
    min_green: int = 10
    yellow_duration: int = 3


def action_to_phase(action: int, plan: PhasePlan) -> int:
    if action not in plan.green_phases:
        raise ValueError(f"Invalid action {action}; valid={sorted(plan.green_phases)}")
    return plan.green_phases[action]


def transition_phase(current_action: int, next_action: int, plan: PhasePlan) -> tuple[int | None, int]:
    if current_action == next_action:
        return None, action_to_phase(next_action, plan)
    return plan.yellow_phases.get((current_action, next_action)), action_to_phase(next_action, plan)
