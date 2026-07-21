from __future__ import annotations

import pickle
import random
from dataclasses import dataclass, field

from src.agents.q_learning import THESIS_ACTIONS, thesis_state


@dataclass
class LinearFunctionApproxQAgent:
    """Chapter 3 Q-learning with Linear Function Approximation (QLF).

    Thesis Chapter 3, QLF equation: Q_theta(x,a)=f(x,a)^T theta.
    Feature order is exactly [q_N,q_S,q_E,q_W,G_N,G_S,G_E,G_W], where
    G_i(a)=0.001 g_i(a). Theta has length 8.
    """

    actions: tuple[tuple[int, int, int, int], ...] = THESIS_ACTIONS
    alpha: float = 0.05
    gamma: float = 0.95
    epsilon: float = 1.0
    theta: list[float] = field(default_factory=lambda: [0.0] * 8)

    def features(self, state, action: tuple[int, int, int, int]) -> list[float]:
        if action not in self.actions:
            raise ValueError(f"invalid Chapter 3 action {action}")
        # Thesis Chapter 3 Simulation: q constants are 0.0005,0.001,0.002;
        # action features are G_i(a)=0.001 g_i(a), preserving N,S,E,W order.
        return list(thesis_state(state)) + [0.001 * float(g) for g in action]

    def q_value(self, state, action) -> float:
        return sum(w * f for w, f in zip(self.theta, self.features(state, action)))

    def act(self, state, evaluate: bool = False):
        if (not evaluate) and random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.actions, key=lambda a: self.q_value(state, a))

    def update(self, state, action, reward: float, next_state, done: bool) -> None:
        # Thesis Chapter 3 QLF update equation.
        feats = self.features(state, action)
        target = reward if done else reward + self.gamma * max(self.q_value(next_state, b) for b in self.actions)
        error = target - self.q_value(state, action)
        for i, feat in enumerate(feats):
            self.theta[i] += self.alpha * feat * error


@dataclass
class PiecewiseLinearQAgent:
    """Chapter 3 Q-learning with Piecewise Linear Function Approximation (QPLF).

    Thesis Chapter 3 QPLF in QLF form: Q_theta(x,a)=f'(x,a)^T theta,
    theta in R^(4n). For action a_j, only block j contains
    [q_N,q_S,q_E,q_W]; all other action blocks are zero. No action features are
    used in QPLF.
    """

    actions: tuple[tuple[int, int, int, int], ...] = THESIS_ACTIONS
    alpha: float = 0.05
    gamma: float = 0.95
    epsilon: float = 1.0
    theta: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.theta:
            self.theta = [0.0] * (4 * len(self.actions))

    def state_features(self, state) -> list[float]:
        values = list(state)[:4]
        values.extend([0.0] * (4 - len(values)))
        return list(thesis_state(values))

    def features(self, state, action) -> list[float]:
        if action not in self.actions:
            raise ValueError(f"invalid Chapter 3 action {action}")
        # Thesis Chapter 3 formal QPLF feature definition y_{4j-3..4j}.
        out = [0.0] * (4 * len(self.actions))
        offset = self.actions.index(action) * 4
        out[offset : offset + 4] = self.state_features(state)
        return out

    def q_value(self, state, action) -> float:
        return sum(w * f for w, f in zip(self.theta, self.features(state, action)))

    def act(self, state, evaluate: bool = False):
        if (not evaluate) and random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.actions, key=lambda action: self.q_value(state, action))

    def update(self, state, action, reward: float, next_state, done: bool) -> None:
        # Thesis Chapter 3 QPLF update equation; only active action block changes.
        feats = self.features(state, action)
        target = reward if done else reward + self.gamma * max(self.q_value(next_state, a) for a in self.actions)
        error = target - self.q_value(state, action)
        for idx, feat in enumerate(feats):
            self.theta[idx] += self.alpha * error * feat

    def save(self, path) -> None:
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
