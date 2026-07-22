#!/usr/bin/env python
"""Run the generated Chapter 4/5 thesis SUMO networks and store thesis-named outputs.

This script is for real SUMO runs only. It does not use the deterministic fallback
simulator or the generated smoke grid from ``run_sumo_experiments.py``. The outputs
are deliberately named with the ``thesis_`` prefix so they are easy to distinguish
from smoke/fallback artifacts.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_thesis_sumo_networks import main as build_thesis_networks
from src.environment.sumo_env import check_sumo_availability


SCENARIOS = [
    {
        "network": "two_intersection",
        "scenario": "chapter4_two_intersection",
        "config": Path("sumo/thesis_ch4_ch5/two_intersection/two_intersection.sumocfg"),
    },
    {
        "network": "nine_intersection",
        "scenario": "chapter4_5_nine_scenario1_low_demand",
        "config": Path("sumo/thesis_ch4_ch5/nine_intersection/scenario1.sumocfg"),
    },
    {
        "network": "nine_intersection",
        "scenario": "chapter5_nine_scenario2_high_demand",
        "config": Path("sumo/thesis_ch4_ch5/nine_intersection/scenario2.sumocfg"),
    },
]


def _float_attr(element: ET.Element, name: str) -> float | None:
    value = element.attrib.get(name)
    return None if value is None else float(value)


def summarize_tripinfo(tripinfo_file: Path) -> dict[str, float | int | None]:
    """Summarize completed-vehicle metrics from a SUMO tripinfo XML file."""
    root = ET.parse(tripinfo_file).getroot()
    tripinfos = root.findall("tripinfo")
    durations = [_float_attr(t, "duration") for t in tripinfos]
    durations = [d for d in durations if d is not None]
    waiting_times = [_float_attr(t, "waitingTime") for t in tripinfos]
    waiting_times = [w for w in waiting_times if w is not None]
    waiting_counts = [_float_attr(t, "waitingCount") for t in tripinfos]
    waiting_counts = [w for w in waiting_counts if w is not None]
    completed = len(tripinfos)
    total_stop_events = int(sum(waiting_counts)) if waiting_counts else None
    return {
        "completed_vehicles": completed,
        "mean_travel_time_seconds": None if not durations else sum(durations) / len(durations),
        "mean_waiting_time_seconds": None if not waiting_times else sum(waiting_times) / len(waiting_times),
        "total_stop_events": total_stop_events,
        "average_stops_per_completed_vehicle": None if not completed or total_stop_events is None else total_stop_events / completed,
    }


def run_sumo_config(sumo_binary: str, config: Path, output_prefix: Path) -> dict[str, object]:
    """Run one thesis SUMO config and return metrics from tripinfo output."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_file = output_prefix.with_name(output_prefix.name + "_summary.xml")
    tripinfo_file = output_prefix.with_name(output_prefix.name + "_tripinfo.xml")
    subprocess.check_call([
        sumo_binary,
        "-c", str(config),
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--summary-output", str(summary_file),
        "--tripinfo-output", str(tripinfo_file),
    ])
    metrics = summarize_tripinfo(tripinfo_file)
    metrics.update({
        "sumo_config": str(config),
        "summary_output": str(summary_file),
        "tripinfo_output": str(tripinfo_file),
    })
    return metrics


def main() -> None:
    availability = check_sumo_availability()
    if not availability.sumo:
        print(json.dumps({"status": "blocked", "errors": availability.errors}, indent=2))
        raise SystemExit(2)

    # Ensure the thesis .net.xml files exist before running the .sumocfg files.
    build_thesis_networks()

    outdir = Path("results/raw")
    rows: list[dict[str, object]] = []
    for item in SCENARIOS:
        prefix = outdir / f"thesis_{item['network']}_{item['scenario']}"
        row = run_sumo_config(availability.sumo, item["config"], prefix)
        row.update({"network": item["network"], "scenario": item["scenario"]})
        rows.append(row)

    csv_file = outdir / "thesis_sumo_metrics.csv"
    json_file = outdir / "thesis_sumo_metrics.json"
    with open(csv_file, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    json_file.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"metrics_csv": str(csv_file), "metrics_json": str(json_file), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
