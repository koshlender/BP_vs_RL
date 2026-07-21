from __future__ import annotations
import random
from src.metrics.stops import StopCounter

class QueueNetworkEnv:
    """Deterministic queue simulator used when SUMO/TraCI is unavailable; mirrors Ch.4/5 APIs."""
    def __init__(self, duration=600, seed=7, demand=None):
        self.duration=duration; self.seed=seed; self.demand=demand or [0.22,0.18,0.20,0.16]; self.t=0
        self.rng=random.Random(seed); self.queues=[0.0]*4; self.stops=StopCounter(); self.completed=0
    def reset(self):
        self.t=0; self.rng.seed(self.seed); self.queues=[0.0]*4; self.stops=StopCounter(); self.completed=0; return list(self.queues)
    def step(self, action:int):
        self.queues=[q+(1 if self.rng.random()<p else 0) for q,p in zip(self.queues,self.demand)]
        served=[0,1] if action==0 else [2,3]; departed=0
        for i in served:
            s=min(self.queues[i],1.0); self.queues[i]-=s; departed+=int(s)
        for vidx,q in enumerate(self.queues): self.stops.update(f'queue{vidx}', 0.0 if q>0 else 5.0)
        self.completed+=departed; self.t+=1
        return list(self.queues), -sum(self.queues), self.t>=self.duration, {'departed':departed,'queue_sum':sum(self.queues)}
    def metrics(self):
        m=self.stops.summary(); m.update({'completed_vehicles':self.completed,'average_travel_time_seconds': 120+10*(sum(self.queues)/len(self.queues))}); return m
