from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from src.thesis.chapter_constants import THESIS_AVAILABLE_GREEN_TIME_SECONDS


def queue_dynamics(
    current_queues: Sequence[float],
    saturation_flows: Sequence[float],
    turn_probabilities: Sequence[Sequence[float]],
) -> list[float]:
    """Thesis Chapter 5, Queue Dynamics.

    Q_i(t+1)=Q_i(t)-sum_a p_ia sigma_i + sum_b p_bi sigma_b.
    Rows of turn_probabilities are p_{outgoing from i to a}; columns are arrivals
    from b to i. This function implements that equation directly for one cycle.
    """
    q = [float(x) for x in current_queues]
    sigma = [float(x) for x in saturation_flows]
    p = [[float(x) for x in row] for row in turn_probabilities]
    if len(q) != len(sigma) or len(p) != len(q) or any(len(row) != len(q) for row in p):
        raise ValueError("queues, saturation flows, and turning matrix must share the same dimension")
    next_queues = []
    for i, q_i in enumerate(q):
        departures = sum(p[i][a] * sigma[i] for a in range(len(q)))
        arrivals = sum(p[b][i] * sigma[b] for b in range(len(q)))
        next_queues.append(q_i - departures + arrivals)
    return next_queues


def phase_weight(
    phase: Sequence[float],
    incoming_queues: Sequence[float],
    outgoing_queues: Sequence[float],
    turn_probabilities: Sequence[Sequence[float]],
) -> float:
    """Thesis Chapter 5, Queue Backpressure Policy weight w_sigma(t)."""
    sigma = [float(x) for x in phase]
    incoming = [float(x) for x in incoming_queues]
    outgoing = [float(x) for x in outgoing_queues]
    p = [[float(x) for x in row] for row in turn_probabilities]
    if len(sigma) != len(incoming) or len(p) != len(incoming) or any(len(row) != len(outgoing) for row in p):
        raise ValueError("phase, incoming queues, outgoing queues, and turning matrix dimensions are inconsistent")
    weight = 0.0
    for i, sigma_i in enumerate(sigma):
        downstream_pressure = sum(p[i][out_idx] * outgoing[out_idx] for out_idx in range(len(outgoing)))
        weight += sigma_i * (incoming[i] - downstream_pressure)
    return weight


def phase_weights(
    phases: Sequence[Sequence[float]],
    incoming_queues: Sequence[float],
    outgoing_queues: Sequence[float],
    turn_probabilities: Sequence[Sequence[float]],
) -> list[float]:
    return [phase_weight(phase, incoming_queues, outgoing_queues, turn_probabilities) for phase in phases]


def max_pressure_phase_index(weights: Sequence[float]) -> int:
    """Thesis Chapter 5, sigma(t)=argmax_sigma w_sigma(t)."""
    if not weights:
        raise ValueError("at least one phase weight is required")
    return max(range(len(weights)), key=lambda idx: float(weights[idx]))


def cyclic_backpressure_proportions(weights: Iterable[float], eta: float) -> list[float]:
    """Thesis Chapter 5, P_sigma=exp(eta w_sigma)/sum exp(eta w_sigma')."""
    w = [float(x) for x in weights]
    if not w:
        raise ValueError("at least one phase weight is required")
    # Numerically stable softmax; mathematically equivalent to the thesis equation.
    max_weight = max(w)
    exp_values = [math.exp(float(eta) * (weight - max_weight)) for weight in w]
    denominator = sum(exp_values)
    return [value / denominator for value in exp_values]


def cyclic_backpressure_green_times(
    weights: Iterable[float],
    eta: float,
    available_green_time: float = THESIS_AVAILABLE_GREEN_TIME_SECONDS,
) -> list[float]:
    """Thesis Chapter 5, cyclic allocation sigma(t)=P_sigma(T-Y)."""
    proportions = cyclic_backpressure_proportions(weights, eta)
    return [p * float(available_green_time) for p in proportions]


def deterministic_integer_durations(durations: Sequence[float], total_duration: int) -> list[int]:
    """Convert fractional executable durations to deterministic integer seconds.

    This is an implementation mechanism for simulators requiring integer-second
    durations, not a separate thesis equation. It floors each non-negative
    duration, then distributes the remaining seconds by largest fractional
    remainder with deterministic index-order tie breaking, so the final sum is
    exactly total_duration.
    """
    raw = [float(duration) for duration in durations]
    if any(duration < 0 for duration in raw):
        raise ValueError("green durations must be non-negative")
    floors = [math.floor(duration) for duration in raw]
    remainder = int(total_duration) - sum(floors)
    if remainder < 0:
        raise ValueError("floored durations exceed requested total duration")
    ranked = sorted(range(len(raw)), key=lambda idx: (-(raw[idx] - floors[idx]), idx))
    out = list(floors)
    for idx in ranked[:remainder]:
        out[idx] += 1
    return out


@dataclass(frozen=True)
class CyclicBackpressureDecision:
    weights: list[float]
    max_pressure_phase: int
    proportions: list[float]
    green_times: list[float]
    executable_green_times: list[int]


def cyclic_backpressure_decision(
    phases: Sequence[Sequence[float]],
    incoming_queues: Sequence[float],
    outgoing_queues: Sequence[float],
    turn_probabilities: Sequence[Sequence[float]],
    eta: float,
    available_green_time: float = THESIS_AVAILABLE_GREEN_TIME_SECONDS,
) -> CyclicBackpressureDecision:
    """Chapter 5 cyclic queue backpressure decision for one intersection/cycle."""
    weights = phase_weights(phases, incoming_queues, outgoing_queues, turn_probabilities)
    proportions = cyclic_backpressure_proportions(weights, eta)
    green_times = [p * float(available_green_time) for p in proportions]
    return CyclicBackpressureDecision(
        weights=weights,
        max_pressure_phase=max_pressure_phase_index(weights),
        proportions=proportions,
        green_times=green_times,
        executable_green_times=deterministic_integer_durations(green_times, int(available_green_time)),
    )
