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


def _output_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _check_call_accepting_colab_sigsegv(cmd: list[str], expected_output: Path) -> None:
    """Run a SUMO tool, accepting Colab SIGSEGV only if output was written.

    Some Colab SUMO packages print `Success.` and create the requested output,
    then exit with SIGSEGV (-11). For this smoke-test network generator, that is
    usable as long as the expected `.net.xml` exists and is non-empty.
    """
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        if exc.returncode == -11 and _output_exists(expected_output):
            return
        raise


def _write_manual_grid_xml(base: Path, grid_number: int, lane_number: int, length_m: int) -> tuple[Path, Path]:
    """Write simple node/edge XML files for a grid, avoiding netgenerate."""
    nodes_file = base.with_suffix(".nod.xml")
    edges_file = base.with_suffix(".edg.xml")
    node_lines = ["<nodes>"]
    for x in range(grid_number):
        for y in range(grid_number):
            node_lines.append(f'  <node id="n{x}_{y}" x="{x * length_m}" y="{y * length_m}" type="traffic_light"/>')
    node_lines.append("</nodes>")
    edge_lines = ["<edges>"]
    for x in range(grid_number):
        for y in range(grid_number):
            if x + 1 < grid_number:
                edge_lines.append(f'  <edge id="e_n{x}_{y}_n{x+1}_{y}" from="n{x}_{y}" to="n{x+1}_{y}" priority="1" numLanes="{lane_number}" speed="13.9"/>')
                edge_lines.append(f'  <edge id="e_n{x+1}_{y}_n{x}_{y}" from="n{x+1}_{y}" to="n{x}_{y}" priority="1" numLanes="{lane_number}" speed="13.9"/>')
            if y + 1 < grid_number:
                edge_lines.append(f'  <edge id="e_n{x}_{y}_n{x}_{y+1}" from="n{x}_{y}" to="n{x}_{y+1}" priority="1" numLanes="{lane_number}" speed="13.9"/>')
                edge_lines.append(f'  <edge id="e_n{x}_{y+1}_n{x}_{y}" from="n{x}_{y+1}" to="n{x}_{y}" priority="1" numLanes="{lane_number}" speed="13.9"/>')
    edge_lines.append("</edges>")
    nodes_file.write_text("\n".join(node_lines) + "\n", encoding="utf-8")
    edges_file.write_text("\n".join(edge_lines) + "\n", encoding="utf-8")
    return nodes_file, edges_file


def generate_grid_network(output_net: Path, grid_number: int = 3, lane_number: int = 1, length_m: int = 500) -> None:
    """Generate a small SUMO grid network for smoke tests.

    Prefer netgenerate with the current `--grid.length` option. Some Colab SUMO
    packages still segfault in netgenerate after printing `Success`; if both
    netgenerate attempts fail, write plain node/edge XML and build the network
    with netconvert instead.
    """
    availability = require_sumo()
    output_net.parent.mkdir(parents=True, exist_ok=True)
    base_cmd = [
        availability.netgenerate or "netgenerate",
        "--grid",
        "--grid.number", str(grid_number),
        "--default.lanenumber", str(lane_number),
        "--tls.guess", "true",
        "--tls.default-type", "static",
        "--output-file", str(output_net),
    ]
    commands = [
        base_cmd[:4] + ["--grid.length", str(length_m)] + base_cmd[4:],
        base_cmd,
    ]
    errors: list[str] = []
    for cmd in commands:
        try:
            _check_call_accepting_colab_sigsegv(cmd, output_net)
            return
        except subprocess.CalledProcessError as exc:
            errors.append(f"{cmd!r} exited with {exc.returncode}")

    netconvert = shutil.which("netconvert")
    if not netconvert:
        raise RuntimeError("Unable to generate SUMO grid network with netgenerate and netconvert is not on PATH. Attempts:\n" + "\n".join(errors))
    nodes_file, edges_file = _write_manual_grid_xml(output_net.with_suffix(""), grid_number, lane_number, length_m)
    netconvert_cmd = [
        netconvert,
        "--node-files", str(nodes_file),
        "--edge-files", str(edges_file),
        "--tls.guess", "true",
        "--output-file", str(output_net),
    ]
    try:
        _check_call_accepting_colab_sigsegv(netconvert_cmd, output_net)
    except subprocess.CalledProcessError as exc:
        errors.append(f"{netconvert_cmd!r} exited with {exc.returncode}")
        raise RuntimeError("Unable to generate SUMO grid network. Attempts:\n" + "\n".join(errors)) from exc


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


class SumoPlatoonTracker:
    """Vehicle-ID platoon progression tracker for real TraCI runs.

    A vehicle is eligible once its routed edge list has at least two normal edges. The
    first two route edges are treated as a corridor smoke-test pair unless a thesis
    corridor-specific integration supplies different edge IDs in future work.
    """
    def __init__(self, stop_speed_threshold: float = 0.1):
        self.stop_speed_threshold = stop_speed_threshold
        self.records: dict[str, dict] = {}

    def observe_vehicle(self, traci_module, veh_id: str, now: float) -> None:
        road_id = traci_module.vehicle.getRoadID(veh_id)
        route = [edge for edge in traci_module.vehicle.getRoute(veh_id) if edge and not edge.startswith(":" )]
        if len(route) < 2:
            return
        rec = self.records.setdefault(veh_id, {
            "first_edge": route[0],
            "second_edge": route[1],
            "first_seen_time": None,
            "second_seen_time": None,
            "stopped_between": False,
            "cleared_second": False,
        })
        if road_id == rec["first_edge"] and rec["first_seen_time"] is None:
            rec["first_seen_time"] = now
        if rec["first_seen_time"] is not None and rec["second_seen_time"] is None:
            if traci_module.vehicle.getSpeed(veh_id) <= self.stop_speed_threshold:
                rec["stopped_between"] = True
        if road_id == rec["second_edge"] and rec["first_seen_time"] is not None:
            rec["second_seen_time"] = now
        if rec["second_seen_time"] is not None and road_id != rec["second_edge"]:
            rec["cleared_second"] = True

    def mark_arrived(self, veh_id: str) -> None:
        if veh_id in self.records and self.records[veh_id]["second_seen_time"] is not None:
            self.records[veh_id]["cleared_second"] = True

    def summary(self) -> dict:
        eligible = [rec for rec in self.records.values() if rec["first_seen_time"] is not None]
        progressed = [rec for rec in eligible if rec["cleared_second"] and not rec["stopped_between"]]
        return {
            "platoon_eligible_vehicles": len(eligible),
            "platoon_progressed_vehicles": len(progressed),
            "platoon_progression_percentage": None if not eligible else 100.0 * len(progressed) / len(eligible),
        }

def run_sumo_policy(config_file: Path, policy: str, duration: int = 600) -> dict:
    availability = require_sumo()
    import traci

    traci.start([availability.sumo or "sumo", "-c", str(config_file), "--no-step-log", "true", "--no-warnings", "true"])
    departed: dict[str, float] = {}
    stop_state: dict[str, bool] = {}
    total_stops = 0
    arrived_travel_times: list[float] = []
    platoon = SumoPlatoonTracker()
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
                platoon.observe_vehicle(traci, veh_id, now)
                speed = traci.vehicle.getSpeed(veh_id)
                if speed <= 0.1 and not stop_state.get(veh_id, False):
                    total_stops += 1
                    stop_state[veh_id] = True
                elif speed >= 1.0:
                    stop_state[veh_id] = False
            for veh_id in traci.simulation.getArrivedIDList():
                platoon.mark_arrived(veh_id)
                if veh_id in departed:
                    arrived_travel_times.append(now - departed[veh_id])
        completed = len(arrived_travel_times)
        result = {
            "policy": policy,
            "completed_vehicles": completed,
            "mean_travel_time_seconds": None if completed == 0 else sum(arrived_travel_times) / completed,
            "total_stop_events": total_stops,
            "average_stops_per_completed_vehicle": None if completed == 0 else total_stops / completed,
        }
        result.update(platoon.summary())
        return result
    finally:
        traci.close(False)
