from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

ZeroQueuePolicy = Literal["zero", "raise", "epsilon"]


@dataclass(frozen=True)
class Chapter4RewardResult:
    """Reward result separating the Chapter 4 equation from zero-queue policy."""

    reward: float
    used_thesis_equation: bool
    zero_queue_policy_applied: bool


def chapter4_queue_reward_result(
    next_queues,
    *,
    zero_queue_policy: ZeroQueuePolicy = "zero",
    zero_queue_reward: float = 0.0,
    epsilon: float = 1.0,
) -> Chapter4RewardResult:
    """Chapter 4 reward R(t)=1/sum_i Q_i(t+1) with explicit zero handling.

    The thesis equation is used exactly whenever the next total queue is
    positive. Chapter 4 does not define what to do when all next queues are zero;
    this function makes that simulation convention explicit and finite by
    default so TD targets cannot become infinite.
    """
    q = [float(queue) for queue in next_queues]
    if any(queue < 0 for queue in q):
        raise ValueError("queue lengths must be non-negative")
    total_queue = sum(q)
    if total_queue > 0:
        return Chapter4RewardResult(1.0 / total_queue, used_thesis_equation=True, zero_queue_policy_applied=False)
    if zero_queue_policy == "raise":
        raise ZeroDivisionError("Chapter 4 reward denominator is zero; thesis provides no zero-queue convention")
    if zero_queue_policy == "epsilon":
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        return Chapter4RewardResult(1.0 / epsilon, used_thesis_equation=False, zero_queue_policy_applied=True)
    if zero_queue_policy != "zero":
        raise ValueError(f"unsupported zero_queue_policy={zero_queue_policy!r}")
    reward = float(zero_queue_reward)
    if not math.isfinite(reward):
        raise ValueError("zero_queue_reward must be finite")
    return Chapter4RewardResult(reward, used_thesis_equation=False, zero_queue_policy_applied=True)


def chapter4_queue_reward(next_queues, **kwargs) -> float:
    """Return only the scalar Chapter 4 reward or zero-queue safeguard value."""
    return chapter4_queue_reward_result(next_queues, **kwargs).reward
