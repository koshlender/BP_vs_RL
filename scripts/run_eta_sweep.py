#!/usr/bin/env python
"""Sweep eta for Chapter 5 cyclic queue backpressure in the fallback simulator."""
from __future__ import annotations
from pathlib import Path
import csv, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.environment.simulated_env import QueueNetworkEnv
from src.rewards.backpressure import cyclic_backpressure_decision
from src.thesis.chapter_constants import THESIS_AVAILABLE_GREEN_TIME_SECONDS, THESIS_YELLOW_TIME_PER_APPROACH_SECONDS
from src.utils.config import load_config, set_seed

SCENARIOS = {
    "scenario1_low_demand": [0.12, 0.10, 0.16, 0.14],
    "scenario2_high_demand": [0.36, 0.28, 0.40, 0.34],
}
ETAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
PHASES = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
IDENTITY_TURNS = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
OUTGOING_QUEUES = [0, 0, 0, 0]


def run_episode(demand: list[float], seed: int, eta: float, duration: int = 600) -> dict:
    env = QueueNetworkEnv(duration=duration, seed=seed, demand=demand)
    queues = env.reset()
    total_queue = 0.0
    cycles = 0
    done = False
    while not done:
        decision = cyclic_backpressure_decision(
            PHASES,
            queues,
            OUTGOING_QUEUES,
            IDENTITY_TURNS,
            eta=eta,
            available_green_time=THESIS_AVAILABLE_GREEN_TIME_SECONDS,
        )
        queues, _reward, done, info = env.step_phase_schedule(
            PHASES,
            decision.executable_green_times,
            yellow_duration_per_phase=THESIS_YELLOW_TIME_PER_APPROACH_SECONDS,
        )
        total_queue += info["queue_sum"]
        cycles += 1
    metrics = env.metrics()
    completed = max(metrics.get("completed_vehicles", 0), 1)
    metrics.update({
        "eta": eta,
        "mean_queue": round(total_queue / max(cycles, 1), 6),
        "mean_delay_seconds": round(total_queue / completed, 6),
        "travel_time_proxy_seconds": round(120 + total_queue / completed, 6),
        "cycles": cycles,
    })
    return metrics


def main() -> None:
    cfg = load_config("configs/chapter_4_5.json")
    base_seed = int(cfg["seed"])
    set_seed(base_seed)
    eval_seeds = [base_seed + 300 + i for i in range(10)]
    rows = []
    best = []
    for scenario, demand in SCENARIOS.items():
        for eta in ETAS:
            eps = [run_episode(demand, seed, eta) for seed in eval_seeds]
            row = {"scenario": scenario, "eta": eta, "eval_episodes": len(eps)}
            for key in ["travel_time_proxy_seconds", "mean_delay_seconds", "average_travel_time_seconds", "completed_vehicles", "total_stop_events", "average_stops_per_observed_vehicle", "mean_queue", "cycles"]:
                vals = [e[key] for e in eps if e[key] is not None]
                row[f"mean_{key}"] = round(sum(vals) / len(vals), 6) if vals else None
            rows.append(row)
        candidates = [r for r in rows if r["scenario"] == scenario]
        best.append(min(candidates, key=lambda r: r["mean_mean_delay_seconds"]))
    outdir = Path("results/raw"); outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "eta_sweep_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    with open(outdir / "eta_sweep_best.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=best[0].keys())
        writer.writeheader(); writer.writerows(best)
    (outdir / "eta_sweep_summary.json").write_text(json.dumps({"rows": rows, "best_by_scenario": best}, indent=2), encoding="utf-8")
    print(json.dumps({"best_by_scenario": best}, indent=2))


if __name__ == "__main__":
    main()
