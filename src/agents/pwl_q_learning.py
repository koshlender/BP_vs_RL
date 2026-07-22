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
    [q_N,q_S,q_E,q_W] in Chapter 3; for Chapter 4 cooperative states the same
    action-block construction is applied to the larger S(t)=[Q(t),Q'(t),a'(t)]
    vector. No action features are used in QPLF.
    """

    actions: tuple[tuple[int, int, int, int], ...] = THESIS_ACTIONS
    alpha: float = 0.05
    gamma: float = 0.95
    epsilon: float = 1.0
    state_dimension: int = 4
    use_chapter3_state_encoding: bool = True
    theta: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.state_dimension <= 0:
            raise ValueError("state_dimension must be positive")
        if not self.theta:
            self.theta = [0.0] * (self.state_dimension * len(self.actions))

    def state_features(self, state) -> list[float]:
        values = list(state)[: self.state_dimension]
        values.extend([0.0] * (self.state_dimension - len(values)))
        if self.use_chapter3_state_encoding:
            values = list(thesis_state(values[:4])) + values[4:]
        total = sum(abs(float(v)) for v in values)
        if total > 1.0:
            values = [float(v) / total for v in values]
        return [float(v) for v in values]

    def features(self, state, action) -> list[float]:
        if action not in self.actions:
            raise ValueError(f"invalid Chapter 3 action {action}")
        # Thesis Chapter 3 formal QPLF feature definition y_{4j-3..4j}.
        out = [0.0] * (self.state_dimension * len(self.actions))
        offset = self.actions.index(action) * self.state_dimension
        out[offset : offset + self.state_dimension] = self.state_features(state)
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
