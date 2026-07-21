#!/usr/bin/env python
"""Run real headless SUMO/TraCI experiments when SUMO is installed.

This script intentionally does not use the deterministic fallback simulator. If SUMO or
TraCI are missing it exits with a clear error so results cannot be mistaken for real
SUMO outputs. The generated SUMO grid is a runnable smoke network, not the original
thesis network; replace it with the thesis `.net.xml`/`.rou.xml` files for final results.
"""
from __future__ import annotations
from pathlib import Path
import csv, json, shutil, subprocess, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.environment.sumo_env import generate_grid_network, run_sumo_policy, write_sumo_config, check_sumo_availability


def write_and_route_trips(net_file: Path, route_file: Path, scenario: str, period: int) -> None:
    import sumolib
    duarouter = shutil.which("duarouter")
    if not duarouter:
        raise RuntimeError("duarouter binary not found on PATH")
    net = sumolib.net.readNet(str(net_file))
    edges = [edge for edge in net.getEdges() if not edge.getFunction() and edge.allows("passenger")]
    if len(edges) < 2:
        raise RuntimeError("generated SUMO network has fewer than two passenger edges")
    west = min(edges, key=lambda e: e.getFromNode().getCoord()[0])
    east = max(edges, key=lambda e: e.getFromNode().getCoord()[0])
    south = min(edges, key=lambda e: e.getFromNode().getCoord()[1])
    north = max(edges, key=lambda e: e.getFromNode().getCoord()[1])
    trips = route_file.with_suffix(".trips.xml")
    lines = ["<routes>", '  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.9"/>']
    tid = 0
    for depart in range(0, 600, period):
        for src, dst, name in [(west, east, "west_east"), (east, west, "east_west"), (north, south, "north_south"), (south, north, "south_north")]:
            lines.append(f'  <trip id="{scenario}_{name}_{tid}" type="car" depart="{depart}" from="{src.getID()}" to="{dst.getID()}"/>')
            tid += 1
    lines.append("</routes>")
    trips.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.check_call([duarouter, "-n", str(net_file), "-r", str(trips), "-o", str(route_file), "--ignore-errors", "true"])


def main() -> None:
    availability = check_sumo_availability()
    if not availability.ok:
        print(json.dumps({"status": "blocked", "errors": availability.errors}, indent=2))
        raise SystemExit(2)
    base = Path("sumo/generated")
    base.mkdir(parents=True, exist_ok=True)
    net = base / "grid3.net.xml"
    generate_grid_network(net)
    rows = []
    for scenario, period in [("scenario1_low_demand", 12), ("scenario2_high_demand", 5)]:
        route = base / f"{scenario}.rou.xml"
        cfg = base / f"{scenario}.sumocfg"
        write_and_route_trips(net, route, scenario, period)
        write_sumo_config(net, route, cfg, end_time=600)
        for policy in ["fixed_alt", "fixed_ns", "pseudo_backpressure"]:
            row = run_sumo_policy(cfg, policy, duration=600)
            row["scenario"] = scenario
            rows.append(row)
    out = Path("results/raw/sumo_experiment_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    Path("results/raw/sumo_experiment_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()
