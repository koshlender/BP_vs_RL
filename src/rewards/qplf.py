from __future__ import annotations

from src.rewards.backpressure import cyclic_backpressure_green_times, phase_weights


def queue_pressure_lyapunov_function(incoming, downstream, turn_matrix=None) -> float:
    """Legacy congestion score retained for reports; not the Chapter 5 policy equation."""
    inc = [float(x) for x in incoming]
    down = [float(x) for x in downstream]
    if turn_matrix is None:
        expected = [down[i % len(down)] for i in range(len(inc))]
    else:
        expected = [sum(float(a) * float(b) for a, b in zip(row, down)) for row in turn_matrix]
    return sum(max(q - e, 0.0) ** 2 for q, e in zip(inc, expected))


def backpressure_weights(incoming, downstream, phases, turn_matrix=None):
    """Compatibility wrapper for Thesis Chapter 5 w_sigma(t)."""
    inc = [float(x) for x in incoming]
    down = [float(x) for x in downstream]
    if turn_matrix is None:
        turn_matrix = [[1.0 if i == j else 0.0 for j in range(len(down))] for i in range(len(inc))]
    return phase_weights(phases, inc, down, turn_matrix)


def cyclic_green_times(weights, cycle_time: float, yellow_time: float, eta: float):
    """Compatibility wrapper for Thesis Chapter 5 sigma(t)=P_sigma(T-Y)."""
    return cyclic_backpressure_green_times(weights, eta, cycle_time - yellow_time)
