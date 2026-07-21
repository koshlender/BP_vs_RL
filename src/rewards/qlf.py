from __future__ import annotations

def queue_length_function(queues) -> float:
    """QLF assumption: sum of squared incoming queues, a Lyapunov-like congestion score."""
    q=[float(x) for x in queues]
    if any(x<0 for x in q): raise ValueError('queues must be non-negative')
    return sum(x*x for x in q)

def reward_from_qlf(previous_queues, current_queues) -> float:
    return queue_length_function(previous_queues)-queue_length_function(current_queues)
