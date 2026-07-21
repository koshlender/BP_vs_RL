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
    def _arrival_step(self):
        self.queues=[q+(1 if self.rng.random()<p else 0) for q,p in zip(self.queues,self.demand)]
    def _record_stops(self):
        for vidx,q in enumerate(self.queues): self.stops.update(f'queue{vidx}', 0.0 if q>0 else 5.0)
    def step(self, action:int):
        self._arrival_step()
        served=[0,1] if action==0 else [2,3]; departed=0
        for i in served:
            s=min(self.queues[i],1.0); self.queues[i]-=s; departed+=int(s)
        self._record_stops()
        self.completed+=departed; self.t+=1
        return list(self.queues), -sum(self.queues), self.t>=self.duration, {'departed':departed,'queue_sum':sum(self.queues)}
    def step_phase_schedule(self, phases, green_durations, yellow_duration_per_phase=4):
        """Execute every Chapter 5 cyclic-backpressure phase for its green duration.

        phases are ordered phase vectors sigma. For each phase, the simulator runs
        the assigned green seconds, serving roads with positive sigma_i, then runs
        the yellow transition after that phase with arrivals but no service. This
        executes the calculated cyclic allocation instead of replacing it with an
        argmax/two-phase fallback.
        """
        if len(phases) != len(green_durations):
            raise ValueError('phases and green durations must have same length')
        departed=0
        for phase, green_seconds in zip(phases, green_durations):
            if green_seconds < 0:
                raise ValueError('green durations must be non-negative')
            for _ in range(int(green_seconds)):
                self._arrival_step()
                for idx, sigma_i in enumerate(phase):
                    if float(sigma_i) > 0:
                        s=min(self.queues[idx], float(sigma_i)); self.queues[idx]-=s; departed+=int(s)
                self._record_stops(); self.t+=1
            for _ in range(int(yellow_duration_per_phase)):
                self._arrival_step(); self._record_stops(); self.t+=1
        self.completed+=departed
        return list(self.queues), -sum(self.queues), self.t>=self.duration, {'departed':departed,'queue_sum':sum(self.queues)}
    def metrics(self):
        m=self.stops.summary(); m.update({'completed_vehicles':self.completed,'average_travel_time_seconds': 120+10*(sum(self.queues)/len(self.queues))}); return m
