from __future__ import annotations
import math

def _matvec(mat, vec): return [sum(float(a)*float(b) for a,b in zip(row, vec)) for row in mat]

def queue_pressure_lyapunov_function(incoming, downstream, turn_matrix=None) -> float:
    """QPLF assumption from Ch.5 pressure term: sum_i max(Q_i - sum_j p_ij Q_j,0)^2."""
    inc=[float(x) for x in incoming]; down=[float(x) for x in downstream]
    exp=_matvec(turn_matrix, down) if turn_matrix is not None else [down[i % len(down)] for i in range(len(inc))]
    return sum(max(q-e,0.0)**2 for q,e in zip(inc,exp))

def backpressure_weights(incoming, downstream, phases, turn_matrix=None):
    inc=[float(x) for x in incoming]; down=[float(x) for x in downstream]
    exp=_matvec(turn_matrix, down) if turn_matrix is not None else [down[i % len(down)] for i in range(len(inc))]
    diff=[q-e for q,e in zip(inc,exp)]
    return [sum(float(p)*d for p,d in zip(phase,diff)) for phase in phases]

def cyclic_green_times(weights, cycle_time: float, yellow_time: float, eta: float):
    m=max(weights); ex=[math.exp(eta*(w-m)) for w in weights]; total=sum(ex)
    return [e/total*(cycle_time-yellow_time) for e in ex]
