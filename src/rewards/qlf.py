from __future__ import annotations


def queue_length_function(queues) -> float:
    """Thesis Chapter 3 reward term: total raw queue sum_i q'_i(t)."""
    q = [float(x) for x in queues]
    if any(x < 0 for x in q):
        raise ValueError("queues must be non-negative")
    return sum(q)


def reward_from_qlf(previous_queues, current_queues) -> float:
    """Thesis Chapter 3 reward equation R(t)=sum_i q'_i(t)-sum_i q'_i(t+1)."""
    return queue_length_function(previous_queues) - queue_length_function(current_queues)
