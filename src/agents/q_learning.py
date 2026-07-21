from __future__ import annotations
import pickle, random
from dataclasses import dataclass, field

@dataclass
class TabularQLearningAgent:
    actions: tuple[int,...]=(0,1); alpha: float=0.1; gamma: float=0.95; epsilon: float=0.1; bins: tuple[int,...]=(0,2,5,10,20,40)
    q: dict[tuple, list[float]]=field(default_factory=dict)
    def discretize(self, state): return tuple(sum(float(x)>b for b in self.bins) for x in state)
    def values(self, state):
        key=self.discretize(state); self.q.setdefault(key, [0.0 for _ in self.actions]); return self.q[key]
    def act(self, state, evaluate=False):
        if (not evaluate) and random.random()<self.epsilon: return random.choice(self.actions)
        vals=self.values(state); return self.actions[max(range(len(vals)), key=lambda i: vals[i])]
    def update(self, s,a,r,ns,done):
        vals=self.values(s); idx=self.actions.index(a); target=r+(0 if done else self.gamma*max(self.values(ns)))
        vals[idx]+=self.alpha*(target-vals[idx])
    def save(self,path):
        with open(path,'wb') as f: pickle.dump(self,f)
    @staticmethod
    def load(path):
        with open(path,'rb') as f: return pickle.load(f)
