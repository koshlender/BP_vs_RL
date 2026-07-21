from __future__ import annotations

import itertools
import pickle
import random
from dataclasses import dataclass, field
from typing import Iterable

from src.thesis.chapter_constants import (
    CH3_GREEN_TIMES_SECONDS,
    CH3_QUEUE_LEVEL_VALUES,
    THESIS_AVAILABLE_GREEN_TIME_SECONDS,
    THESIS_CYCLE_TIME_SECONDS,
    THESIS_TOTAL_YELLOW_TIME_SECONDS,
)

THESIS_QUEUE_LEVELS = CH3_QUEUE_LEVEL_VALUES
THESIS_GREEN_TIMES = CH3_GREEN_TIMES_SECONDS
THESIS_CYCLE_TIME = THESIS_CYCLE_TIME_SECONDS
THESIS_TOTAL_YELLOW_TIME = THESIS_TOTAL_YELLOW_TIME_SECONDS
THESIS_AVAILABLE_GREEN_TIME = THESIS_AVAILABLE_GREEN_TIME_SECONDS
THESIS_QUEUE_LEVELS = (0.0005, 0.001, 0.002)
THESIS_GREEN_TIMES = (8, 16, 24, 32)
THESIS_CYCLE_TIME = 80
THESIS_TOTAL_YELLOW_TIME = 16
THESIS_AVAILABLE_GREEN_TIME = THESIS_CYCLE_TIME - THESIS_TOTAL_YELLOW_TIME


def thesis_actions() -> tuple[tuple[int, int, int, int], ...]:
    """Thesis Chapter 3, Traffic control problem as MDP, Action.

    Action a(t)=[g_N(t),g_S(t),g_E(t),g_W(t)] with each green in
    {8,16,24,32} and sum_j g_j(t)=T-Y=64 seconds.
    """
    return tuple(
        action
        for action in itertools.product(THESIS_GREEN_TIMES, repeat=4)
        if sum(action) == THESIS_AVAILABLE_GREEN_TIME
    )


THESIS_ACTIONS = thesis_actions()


def discretize_queue(queue: float) -> float:
    """Thesis Chapter 3, Traffic control problem as MDP, State.

    q_i(t)=a if q'_i(t)<=25, b if 26<q'_i(t)<=50, c if 50<q'_i(t).
    The thesis later fixes a,b,c as 0.0005, 0.001, 0.002 respectively.

    Ambiguity preserved: the text omits q'_i(t)=26 from the second interval
    (it says 26 < q'_i(t) <= 50). Integer queue lengths at 26 are assigned to
    the middle level to avoid creating an extra state not present in the thesis.
    """
    q = float(queue)
    if q <= 25:
        return THESIS_QUEUE_LEVELS[0]
    if q <= 50:
        return THESIS_QUEUE_LEVELS[1]
    return THESIS_QUEUE_LEVELS[2]


def thesis_state(queues: Iterable[float]) -> tuple[float, float, float, float]:
    """Thesis Chapter 3 state x=[q_N(t),q_S(t),q_E(t),q_W(t)]."""
    values = tuple(discretize_queue(q) for q in queues)
    if len(values) != 4:
        raise ValueError("Chapter 3 isolated-intersection state must have four queues ordered N,S,E,W")
    return values


def throughput_reward(previous_queues: Iterable[float], next_queues: Iterable[float]) -> float:
    """Thesis Chapter 3 reward equation R(t)=sum_i q'_i(t)-sum_i q'_i(t+1)."""
    return sum(float(q) for q in previous_queues) - sum(float(q) for q in next_queues)


@dataclass
class TabularQLearningAgent:
    """Full-State Tabular Q-learning for the Chapter 3 isolated signal.

    State: discretized [q_N,q_S,q_E,q_W]. Action: all green-time vectors in
    {8,16,24,32}^4 with total green T-Y=64. Update: Watkins Q-learning,
    Q_{k+1}=Q_k+alpha[R+gamma max_b Q(Y,b)-Q_k].
    """

    actions: tuple[tuple[int, int, int, int], ...] = THESIS_ACTIONS
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_decay: float = 0.05
    q: dict[tuple[float, float, float, float], list[float]] = field(default_factory=dict)

    def discretize(self, state):
        return thesis_state(state)

    def decay_epsilon(self) -> None:
        """Thesis Chapter 3 Simulation: epsilon starts at 1 and drops 0.05/episode until episode 20."""
        self.epsilon = max(0.0, self.epsilon - self.epsilon_decay)

    def values(self, state):
        key = self.discretize(state)
        self.q.setdefault(key, [0.0 for _ in self.actions])
        return self.q[key]

    def act(self, state, evaluate=False):
        if (not evaluate) and random.random() < self.epsilon:
            return random.choice(self.actions)
        vals = self.values(state)
        return self.actions[max(range(len(vals)), key=lambda i: vals[i])]

    def update(self, s, a, r, ns, done):
        vals = self.values(s)
        idx = self.actions.index(a)
        target = r + (0.0 if done else self.gamma * max(self.values(ns)))
        vals[idx] += self.alpha * (target - vals[idx])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
