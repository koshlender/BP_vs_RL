from __future__ import annotations
from dataclasses import dataclass, field
import pickle, random

@dataclass
class PiecewiseLinearQAgent:
    """QPLF-style Q-learning with action-wise piecewise-linear blocks.

    This implements the thesis feature construction f'(x,a): for n actions and four
    queue features [qN, qS, qE, qW], the active action owns one 4-parameter block
    and all other action blocks are zero. Queue features are L1-normalized so
    sum_k |f'_k(x,a)| <= 1 when queues are non-negative.
    """
    actions: tuple[int, ...] = (0, 1)
    alpha: float = 0.05
    gamma: float = 0.95
    epsilon: float = 0.2
    state_dim: int = 4
    theta: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.theta:
            self.theta = [0.0] * (self.state_dim * len(self.actions))

    def state_features(self, state) -> list[float]:
        raw = [max(0.0, float(x)) for x in list(state)[: self.state_dim]]
        raw.extend([0.0] * (self.state_dim - len(raw)))
        total = sum(abs(x) for x in raw)
        return raw if total <= 1.0 else [x / total for x in raw]

    def features(self, state, action: int) -> list[float]:
        if action not in self.actions:
            raise ValueError(f"invalid action {action}; valid={self.actions}")
        out = [0.0] * (self.state_dim * len(self.actions))
        offset = self.actions.index(action) * self.state_dim
        q = self.state_features(state)
        out[offset : offset + self.state_dim] = q
        return out

    def q_value(self, state, action: int) -> float:
        feats = self.features(state, action)
        return sum(w * f for w, f in zip(self.theta, feats))

    def act(self, state, evaluate: bool = False) -> int:
        if (not evaluate) and random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.actions, key=lambda action: self.q_value(state, action))

    def update(self, state, action: int, reward: float, next_state, done: bool) -> None:
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
