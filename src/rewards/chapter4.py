from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

ZeroQueuePolicy = Literal["zero", "raise"]


@dataclass(frozen=True)
class Chapter4RewardResult:
    """Reward result separating the thesis equation from zero-denominator policy."""

    reward: float
    used_thesis_equation: bool
    zero_queue_policy_applied: bool


def chapter4_queue_reward_result(
    next_queues,
    *,
    zero_queue_policy: ZeroQueuePolicy = "zero",
    zero_queue_reward: float = 0.0,
) -> Chapter4RewardResult:
    """Thesis Chapter 4 reward with explicit zero-queue simulation policy.

    Thesis equation: R(t)=1 / sum_{i in I} Q_i(t+1). The available thesis text
    does not define the zero-denominator case, and returning infinity would make
    TD targets and parameters non-finite. When the denominator is positive this
    function returns the thesis equation exactly. When the denominator is zero,
    the configurable policy is outside the thesis equation; the default finite
    safeguard returns 0.0 and marks that the equation was not used.
    """
    q = [float(queue) for queue in next_queues]
    if any(queue < 0 for queue in q):
        raise ValueError("queue lengths must be non-negative")
    total_queue = sum(q)
    if total_queue > 0:
        reward = 1.0 / total_queue
        return Chapter4RewardResult(reward, used_thesis_equation=True, zero_queue_policy_applied=False)
    if zero_queue_policy == "raise":
        raise ZeroDivisionError("Chapter 4 reward denominator is zero; thesis provides no zero-queue convention")
    if zero_queue_policy != "zero":
        raise ValueError(f"unsupported zero_queue_policy={zero_queue_policy!r}")
    reward = float(zero_queue_reward)
    if not math.isfinite(reward):
        raise ValueError("zero_queue_reward must be finite")
    return Chapter4RewardResult(reward, used_thesis_equation=False, zero_queue_policy_applied=True)


def chapter4_queue_reward(next_queues, **kwargs) -> float:
    """Return only the scalar Chapter 4 reward / zero-queue safeguard value."""
    return chapter4_queue_reward_result(next_queues, **kwargs).reward
