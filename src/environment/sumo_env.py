from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import shutil, subprocess, sys

@dataclass
class SumoAvailability:
    sumo: str | None
    netgenerate: str | None
    traci_importable: bool
    sumolib_importable: bool
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.sumo and self.netgenerate and self.traci_importable and self.sumolib_importable)


def check_sumo_availability() -> SumoAvailability:
    errors: list[str] = []
    sumo = shutil.which("sumo")
    netgenerate = shutil.which("netgenerate")
    try:
        import traci  # noqa: F401
        traci_importable = True
    except Exception as exc:  # pragma: no cover - environment dependent
        traci_importable = False
        errors.append(f"cannot import traci: {exc!r}")
    try:
        import sumolib  # noqa: F401
        sumolib_importable = True
    except Exception as exc:  # pragma: no cover - environment dependent
        sumolib_importable = False
        errors.append(f"cannot import sumolib: {exc!r}")
    if not sumo:
        errors.append("sumo binary not found on PATH")
    if not netgenerate:
        errors.append("netgenerate binary not found on PATH")
    return SumoAvailability(sumo, netgenerate, traci_importable, sumolib_importable, errors)


def require_sumo() -> SumoAvailability:
    availability = check_sumo_availability()
    if not availability.ok:
        detail = "\n".join(f"- {err}" for err in availability.errors)
        raise RuntimeError(
            "Real SUMO experiments require SUMO command-line tools and Python TraCI bindings.\n"
            f"Current environment is missing:\n{detail}\n"
            "Install system packages such as `sumo sumo-tools` and ensure SUMO_HOME/tools is on PYTHONPATH."
        )
    return availability


def generate_grid_network(output_net: Path, grid_number: int = 3, lane_number: int = 1, length_m: int = 500) -> None:
    availability = require_sumo()
    output_net.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        availability.netgenerate or "netgenerate",
        "--grid",
        "--grid.number", str(grid_number),
        "--default.lanenumber", str(lane_number),
        "--default.length", str(length_m),
        "--tls.guess", "true",
        "--tls.default-type", "static",
        "--output-file", str(output_net),
    ]
    subprocess.check_call(cmd)


def write_sumo_config(net_file: Path, route_file: Path, config_file: Path, end_time: int, step_length: float = 1.0) -> None:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        f'''<configuration>
  <input>
    <net-file value="{net_file.name}"/>
    <route-files value="{route_file.name}"/>
  </input>
  <time>
    <begin value="0"/>
    <end value="{end_time}"/>
    <step-length value="{step_length}"/>
  </time>
</configuration>
''',
        encoding="utf-8",
    )


def run_sumo_policy(config_file: Path, policy: str, duration: int = 600) -> dict:
    availability = require_sumo()
    import traci

    traci.start([availability.sumo or "sumo", "-c", str(config_file), "--no-step-log", "true", "--no-warnings", "true"])
    departed: dict[str, float] = {}
    stop_state: dict[str, bool] = {}
    total_stops = 0
    arrived_travel_times: list[float] = []
    try:
        tls_ids = list(traci.trafficlight.getIDList())
        for step in range(duration):
            for tls_id in tls_ids:
                phases = traci.trafficlight.getAllProgramLogics(tls_id)[0].phases
                green_indices = [idx for idx, phase in enumerate(phases) if "G" in phase.state or "g" in phase.state]
                if green_indices:
                    if policy == "fixed_ns":
                        phase_idx = green_indices[0]
                    elif policy == "fixed_alt":
                        phase_idx = green_indices[(step // 30) % len(green_indices)]
                    else:
                        phase_idx = green_indices[(step // 15 + len(tls_id)) % len(green_indices)]
                    traci.trafficlight.setPhase(tls_id, phase_idx)
            traci.simulationStep()
            now = traci.simulation.getTime()
            for veh_id in traci.simulation.getDepartedIDList():
                departed[veh_id] = now
                stop_state[veh_id] = False
            for veh_id in traci.vehicle.getIDList():
                speed = traci.vehicle.getSpeed(veh_id)
                if speed <= 0.1 and not stop_state.get(veh_id, False):
                    total_stops += 1
                    stop_state[veh_id] = True
                elif speed >= 1.0:
                    stop_state[veh_id] = False
            for veh_id in traci.simulation.getArrivedIDList():
                if veh_id in departed:
                    arrived_travel_times.append(now - departed[veh_id])
        completed = len(arrived_travel_times)
        return {
            "policy": policy,
            "completed_vehicles": completed,
            "mean_travel_time_seconds": None if completed == 0 else sum(arrived_travel_times) / completed,
            "total_stop_events": total_stops,
            "average_stops_per_completed_vehicle": None if completed == 0 else total_stops / completed,
        }
    finally:
        traci.close(False)
