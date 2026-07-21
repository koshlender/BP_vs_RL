from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class VehicleStopState:
    stopped_time: float=0.0; counted: bool=False; observed: bool=True; completed: bool=False

@dataclass
class StopCounter:
    stop_speed_threshold: float=0.1; minimum_stop_duration: float=2.0; reset_speed_threshold: float=1.0
    states: dict[str, VehicleStopState]=field(default_factory=dict); total_stop_events: int=0
    def update(self, veh_id: str, speed: float, dt: float=1.0) -> None:
        s=self.states.setdefault(veh_id, VehicleStopState())
        if speed <= self.stop_speed_threshold:
            s.stopped_time += dt
            if not s.counted and s.stopped_time >= self.minimum_stop_duration:
                self.total_stop_events += 1; s.counted=True
        elif speed >= self.reset_speed_threshold:
            s.stopped_time=0.0; s.counted=False
    def mark_completed(self, veh_id: str) -> None:
        self.states.setdefault(veh_id, VehicleStopState()).completed=True
    def summary(self) -> dict:
        observed=len(self.states); completed=sum(v.completed for v in self.states.values())
        return {'total_stop_events':self.total_stop_events,'observed_vehicles':observed,'completed_vehicles':completed,
                'average_stops_per_observed_vehicle': None if observed==0 else self.total_stop_events/observed,
                'average_stops_per_completed_vehicle': None if completed==0 else self.total_stop_events/completed}
