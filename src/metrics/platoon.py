from __future__ import annotations
from dataclasses import dataclass, field
import statistics

@dataclass
class PlatoonVehicle:
    route: str; first_time: float|None=None; second_time: float|None=None; clear_time: float|None=None
    stopped_between: bool=False; stopped_at_second: bool=False; disappeared: bool=False

@dataclass
class PlatoonProgressionTracker:
    eligible_routes: set[str]; stop_speed_threshold: float=0.1; count_second_signal_stop_as_failure: bool=True; max_corridor_travel_time: float=300
    vehicles: dict[str, PlatoonVehicle]=field(default_factory=dict)
    def cross_first(self, veh_id: str, route: str, time: float):
        if route in self.eligible_routes: self.vehicles[veh_id]=PlatoonVehicle(route, first_time=time)
    def observe_corridor(self, veh_id: str, speed: float, at_second: bool=False):
        v=self.vehicles.get(veh_id)
        if not v: return
        if speed <= self.stop_speed_threshold:
            if at_second: v.stopped_at_second=True
            else: v.stopped_between=True
    def reach_second(self, veh_id: str, time: float):
        if veh_id in self.vehicles: self.vehicles[veh_id].second_time=time
    def clear_second(self, veh_id: str, time: float):
        if veh_id in self.vehicles: self.vehicles[veh_id].clear_time=time
    def disappear(self, veh_id: str):
        if veh_id in self.vehicles: self.vehicles[veh_id].disappeared=True
    def summary(self) -> dict:
        eligible=[v for v in self.vehicles.values() if v.first_time is not None and not v.disappeared]
        completed=[v for v in eligible if v.clear_time is not None and (v.clear_time-v.first_time)<=self.max_corridor_travel_time]
        def ok(v): return not v.stopped_between and not (self.count_second_signal_stop_as_failure and v.stopped_at_second)
        progressed=[v for v in completed if ok(v)]
        times=[v.clear_time-v.first_time for v in completed]
        return {'eligible_vehicles':len(eligible),'vehicles_progressing_without_stopping':len(progressed),
                'platoon_progression_percentage': None if not eligible else 100*len(progressed)/len(eligible),
                'mean_corridor_travel_time': None if not times else statistics.mean(times),
                'median_corridor_travel_time': None if not times else statistics.median(times),
                'stops_between_intersections':sum(v.stopped_between for v in eligible),'stops_at_second_intersection':sum(v.stopped_at_second for v in eligible)}
