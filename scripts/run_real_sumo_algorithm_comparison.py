#!/usr/bin/env python
"""Run the requested Chapter 4/5 algorithms on real SUMO via TraCI.

This is the real-SUMO counterpart to ``run_thesis_algorithm_comparison.py``. It
starts SUMO for each scenario/episode, controls traffic lights through TraCI, and
stores actual SUMO trip-time metrics. It intentionally does not use the compact
queue simulator.

Implemented real-SUMO policies:
* Independent Learner - Full RL (tabular local state)
* Independent Learner - QPLF (piecewise-linear local state)
* Semi-Coordinated - Full RL (tabular local + neighboring aggregate state)
* Semi-Coordinated - QPLF (piecewise-linear local + neighboring aggregate state)
* Cyclic Queue Backpressure (eta sweep 0.1..1.2; best eta used for comparison)

The generated thesis SUMO network is a reconstruction from the thesis tables and
figures available in this repository. If you have the author's original SUMO
network/route files, point this script at those files for final thesis-grade runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import inspect
import math
import random
import sys
from pathlib import Path
from typing import Iterable

TRACI_START_RETRIES = 3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_thesis_sumo_networks import main as build_thesis_networks
from scripts.run_thesis_algorithm_comparison import FlexibleTabularAgent
from src.agents.pwl_q_learning import PiecewiseLinearQAgent
from src.rewards.chapter4 import chapter4_queue_reward
from src.environment.sumo_env import check_sumo_availability
from src.utils.config import load_config, set_seed

RL_POLICIES = [
    "independent_full_rl",
    "independent_qplf",
    "semi_coordinated_full_rl",
    "semi_coordinated_qplf",
]
BACKPRESSURE_POLICY = "cyclic_queue_backpressure"
ALL_POLICIES = RL_POLICIES + [BACKPRESSURE_POLICY]
ETAS = [round(i / 10, 1) for i in range(1, 13)]
POLICY_LABELS = {
    "independent_full_rl": "Independent Learner - Full RL",
    "independent_qplf": "Independent Learner - QPLF",
    "semi_coordinated_full_rl": "Semi-Coordinated - Full RL",
    "semi_coordinated_qplf": "Semi-Coordinated - QPLF",
    BACKPRESSURE_POLICY: "Cyclic Queue Backpressure",
}
SCENARIOS = {
    "chapter4_two_intersection": Path("sumo/thesis_ch4_ch5/two_intersection/two_intersection.sumocfg"),
    "chapter4_5_nine_scenario1_low_demand": Path("sumo/thesis_ch4_ch5/nine_intersection/scenario1.sumocfg"),
    "chapter5_nine_scenario2_high_demand": Path("sumo/thesis_ch4_ch5/nine_intersection/scenario2.sumocfg"),
}


def parse_eta_values(text: str) -> list[float]:
    try:
        values = [round(float(item.strip()), 3) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("eta values must be numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("provide at least one eta value")
    return values

def softmax(values: list[float], eta: float) -> list[float]:
    if not values:
        return []
    scaled = [eta * value for value in values]
    offset = max(scaled)
    exps = [math.exp(value - offset) for value in scaled]
    denom = sum(exps)
    return [value / denom for value in exps]


def integer_durations(proportions: list[float], total: int) -> list[int]:
    if not proportions:
        return []
    raw = [p * total for p in proportions]
    floors = [int(math.floor(x)) for x in raw]
    remaining = total - sum(floors)
    order = sorted(range(len(raw)), key=lambda idx: raw[idx] - floors[idx], reverse=True)
    for idx in order[:remaining]:
        floors[idx] += 1
    return floors


def make_agent(policy: str, action_count: int):
    actions = tuple(range(action_count))
    if policy in {"independent_full_rl", "semi_coordinated_full_rl"}:
        return FlexibleTabularAgent(actions=actions, alpha=0.1, epsilon=0.35)
    if policy == "independent_qplf":
        return PiecewiseLinearQAgent(actions=actions, alpha=0.05, epsilon=0.35, state_dimension=4, use_chapter3_state_encoding=False)
    if policy == "semi_coordinated_qplf":
        return PiecewiseLinearQAgent(actions=actions, alpha=0.05, epsilon=0.35, state_dimension=6, use_chapter3_state_encoding=False)
    raise ValueError(policy)


def green_phase_indices(traci, tls_id: str) -> list[int]:
    phases = traci.trafficlight.getAllProgramLogics(tls_id)[0].phases
    indices = [idx for idx, phase in enumerate(phases) if "G" in phase.state or "g" in phase.state]
    return indices or [0]


def controlled_lanes(traci, tls_id: str) -> list[str]:
    lanes = []
    for lane in traci.trafficlight.getControlledLanes(tls_id):
        if lane not in lanes:
            lanes.append(lane)
    return lanes


def tls_queue_vector(traci, tls_id: str, width: int = 4) -> list[float]:
    values = [float(traci.lane.getLastStepHaltingNumber(lane)) for lane in controlled_lanes(traci, tls_id)]
    if len(values) < width:
        values.extend([0.0] * (width - len(values)))
    return values[:width]


def semi_state(traci, tls_id: str, previous_action: int) -> list[float]:
    local = tls_queue_vector(traci, tls_id)
    tls_ids = list(traci.trafficlight.getIDList())
    neighbor_queues = [sum(tls_queue_vector(traci, other)) for other in tls_ids if other != tls_id]
    neighbor_mean = 0.0 if not neighbor_queues else sum(neighbor_queues) / len(neighbor_queues)
    return local + [neighbor_mean, float(previous_action)]


def rl_state(traci, policy: str, tls_id: str, previous_action: int) -> list[float]:
    if policy in {"independent_full_rl", "independent_qplf"}:
        return tls_queue_vector(traci, tls_id)
    return semi_state(traci, tls_id, previous_action)


def phase_weight_for_queue(traci, tls_id: str, phase_idx: int) -> float:
    phases = traci.trafficlight.getAllProgramLogics(tls_id)[0].phases
    phase_state = phases[phase_idx].state
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    total = 0.0
    for signal_idx, lane in enumerate(lanes[: len(phase_state)]):
        if phase_state[signal_idx] in {"G", "g"}:
            total += float(traci.lane.getLastStepHaltingNumber(lane))
    return total


def simulation_metrics(arrived_travel_times: list[float], total_stops: int) -> dict[str, float | int | None]:
    durations = arrived_travel_times
    return {
        "completed_vehicles": len(durations),
        "mean_travel_time_seconds": None if not durations else sum(durations) / len(durations),
        "total_stop_events": total_stops,
        "average_stops_per_completed_vehicle": None if not durations else total_stops / len(durations),
    }


def run_rl_episode(traci, policy: str, agents: dict[str, object], duration: int, cycle_seconds: int, train: bool, record_trace: bool) -> tuple[dict[str, object], list[dict[str, object]]]:
    tls_ids = list(traci.trafficlight.getIDList())
    green_indices = {tls_id: green_phase_indices(traci, tls_id) for tls_id in tls_ids}
    previous_actions = {tls_id: 0 for tls_id in tls_ids}
    departed: dict[str, float] = {}
    arrived_travel_times: list[float] = []
    stopped: dict[str, bool] = {}
    total_stops = 0
    queue_area = 0.0
    trace: list[dict[str, object]] = []
    while traci.simulation.getTime() < duration:
        states: dict[str, list[float]] = {}
        actions: dict[str, int] = {}
        for tls_id in tls_ids:
            states[tls_id] = rl_state(traci, policy, tls_id, previous_actions[tls_id])
            action = agents[tls_id].act(states[tls_id], evaluate=not train)
            actions[tls_id] = int(action)
            phase_idx = green_indices[tls_id][int(action) % len(green_indices[tls_id])]
            traci.trafficlight.setPhase(tls_id, phase_idx)
        for _ in range(cycle_seconds):
            if traci.simulation.getTime() >= duration:
                break
            traci.simulationStep()
            now = traci.simulation.getTime()
            for veh_id in traci.simulation.getDepartedIDList():
                departed[veh_id] = now
                stopped[veh_id] = False
            for veh_id in traci.simulation.getArrivedIDList():
                if veh_id in departed:
                    arrived_travel_times.append(now - departed[veh_id])
            for veh_id in traci.vehicle.getIDList():
                speed = traci.vehicle.getSpeed(veh_id)
                if speed <= 0.1 and not stopped.get(veh_id, False):
                    total_stops += 1
                    stopped[veh_id] = True
                elif speed >= 1.0:
                    stopped[veh_id] = False
            qsum = sum(sum(tls_queue_vector(traci, tls_id)) for tls_id in tls_ids)
            queue_area += qsum
            if record_trace:
                trace.append({"time_seconds": now, "policy": policy, "policy_label": POLICY_LABELS[policy], "total_queue": qsum})
        for tls_id in tls_ids:
            next_state = rl_state(traci, policy, tls_id, actions[tls_id])
            reward = chapter4_queue_reward(tls_queue_vector(traci, tls_id), zero_queue_policy="epsilon", epsilon=1.0)
            if train:
                agents[tls_id].update(states[tls_id], actions[tls_id], reward, next_state, False)
            previous_actions[tls_id] = actions[tls_id]
    metrics = simulation_metrics(arrived_travel_times, total_stops)
    metrics.update({
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "mean_network_queue": queue_area / max(duration, 1),
        "queue_area": queue_area,
    })
    return metrics, trace


def run_backpressure_episode(traci, eta: float, duration: int, cycle_seconds: int, record_trace: bool) -> tuple[dict[str, object], list[dict[str, object]]]:
    tls_ids = list(traci.trafficlight.getIDList())
    green_indices = {tls_id: green_phase_indices(traci, tls_id) for tls_id in tls_ids}
    departed: dict[str, float] = {}
    arrived_travel_times: list[float] = []
    stopped: dict[str, bool] = {}
    total_stops = 0
    queue_area = 0.0
    trace: list[dict[str, object]] = []
    while traci.simulation.getTime() < duration:
        schedules: dict[str, list[tuple[int, int]]] = {}
        for tls_id in tls_ids:
            weights = [phase_weight_for_queue(traci, tls_id, phase_idx) for phase_idx in green_indices[tls_id]]
            durations = integer_durations(softmax(weights, eta), cycle_seconds)
            schedules[tls_id] = list(zip(green_indices[tls_id], durations))
        for tick in range(cycle_seconds):
            if traci.simulation.getTime() >= duration:
                break
            for tls_id in tls_ids:
                elapsed = 0
                selected = schedules[tls_id][0][0]
                for phase_idx, seconds in schedules[tls_id]:
                    elapsed += seconds
                    if tick < elapsed:
                        selected = phase_idx
                        break
                traci.trafficlight.setPhase(tls_id, selected)
            traci.simulationStep()
            now = traci.simulation.getTime()
            for veh_id in traci.simulation.getDepartedIDList():
                departed[veh_id] = now
                stopped[veh_id] = False
            for veh_id in traci.simulation.getArrivedIDList():
                if veh_id in departed:
                    arrived_travel_times.append(now - departed[veh_id])
            for veh_id in traci.vehicle.getIDList():
                speed = traci.vehicle.getSpeed(veh_id)
                if speed <= 0.1 and not stopped.get(veh_id, False):
                    total_stops += 1
                    stopped[veh_id] = True
                elif speed >= 1.0:
                    stopped[veh_id] = False
            qsum = sum(sum(tls_queue_vector(traci, tls_id)) for tls_id in tls_ids)
            queue_area += qsum
            if record_trace:
                trace.append({"time_seconds": now, "policy": BACKPRESSURE_POLICY, "policy_label": POLICY_LABELS[BACKPRESSURE_POLICY], "eta": eta, "total_queue": qsum})
    metrics = simulation_metrics(arrived_travel_times, total_stops)
    metrics.update({
        "policy": BACKPRESSURE_POLICY,
        "policy_label": POLICY_LABELS[BACKPRESSURE_POLICY],
        "eta": eta,
        "mean_network_queue": queue_area / max(duration, 1),
        "queue_area": queue_area,
    })
    return metrics, trace



def choose_best_eta(
    current_eta: float | None,
    current_delay: float | None,
    current_queue: float | None,
    candidate_metrics: dict[str, object],
) -> tuple[float | None, float | None, float | None, str | None]:
    """Prefer lowest travel time, falling back to lowest queue for short runs."""
    eta = candidate_metrics.get("eta")
    if eta is None:
        return current_eta, current_delay, current_queue, None
    delay = candidate_metrics.get("mean_travel_time_seconds")
    if delay is not None:
        delay_value = float(delay)
        if current_delay is None or delay_value < current_delay:
            return float(eta), delay_value, current_queue, "mean_travel_time_seconds"
        return current_eta, current_delay, current_queue, "mean_travel_time_seconds" if current_delay is not None else "mean_network_queue"
    if current_delay is not None:
        return current_eta, current_delay, current_queue, "mean_travel_time_seconds"
    queue = candidate_metrics.get("mean_network_queue")
    if queue is None:
        return current_eta, current_delay, current_queue, None
    queue_value = float(queue)
    if current_queue is None or queue_value < current_queue:
        return float(eta), current_delay, queue_value, "mean_network_queue"
    return current_eta, current_delay, current_queue, "mean_network_queue"

def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_simple_svg(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="400"><rect width="100%" height="100%" fill="white"/><text x="30" y="45" font-family="sans-serif" font-size="22">{title}</text><text x="30" y="90" font-family="sans-serif" font-size="14">Use the matching real_sumo_*.csv file for plotted values.</text></svg>\n', encoding="utf-8")


def sumo_command(sumo_binary: str, config: Path) -> list[str]:
    return [sumo_binary, "-c", str(config), "--no-step-log", "true", "--no-warnings", "true"]


def start_traci(traci, sumo_binary: str, config: Path, retries: int = TRACI_START_RETRIES) -> None:
    """Start TraCI with bounded retries and a Colab-friendly failure message.

    TraCI normally prints repeated `Retrying in 1 seconds` lines while it waits
    for SUMO to open the TraCI port. In Colab, a broken SUMO install or a SUMO
    startup crash can leave users watching those retries without the actionable
    command/config context. Keep retries short and re-raise with the exact SUMO
    command to run directly for diagnostics.
    """
    cmd = sumo_command(sumo_binary, config)
    rendered = " ".join(cmd)
    try:
        start_parameters = inspect.signature(traci.start).parameters
    except (TypeError, ValueError):
        start_parameters = {}
    if "numRetries" not in start_parameters:
        raise RuntimeError(
            "The imported TraCI module does not support bounded startup retries. "
            "In Colab, remove mismatched TraCI packages and use SUMO's bundled "
            "tools path first, for example:\n"
            "  pip uninstall -y traci sumolib\n"
            "  PYTHONPATH=/usr/share/sumo/tools:$PYTHONPATH python -c "
            "'import traci; print(traci.__file__)'\n"
            f"After fixing TraCI, rerun this SUMO command directly if startup "
            f"still fails:\n{rendered}"
        )
    try:
        traci.start(cmd, numRetries=retries)
    except Exception as exc:
        raise RuntimeError(
            "TraCI could not connect to SUMO after "
            f"{retries} retries. In Colab this usually means the SUMO process "
            "crashed before opening its TraCI port or the installed SUMO/TraCI "
            "versions are incompatible. First run this direct SUMO check in a "
            f"notebook cell:\n{rendered} --end 1\n"
            "If direct SUMO works, force Colab to use SUMO's bundled Python "
            "tools before any pip-installed TraCI package:\n"
            "  pip uninstall -y traci sumolib\n"
            "  PYTHONPATH=.:/usr/share/sumo/tools:$PYTHONPATH python "
            "scripts/run_real_sumo_algorithm_comparison.py --scenario "
            "chapter5_nine_scenario2_high_demand --episodes 1 --duration 60"
        ) from exc


def run_one_config(sumo_binary: str, config: Path, policy: str, episode: int, duration: int, cycle_seconds: int, agents_by_tls: dict[str, object] | None = None, eta: float | None = None, record_trace: bool = False):
    import traci
    start_traci(traci, sumo_binary, config)
    try:
        if policy == BACKPRESSURE_POLICY:
            assert eta is not None
            return run_backpressure_episode(traci, eta, duration, cycle_seconds, record_trace)
        assert agents_by_tls is not None
        return run_rl_episode(traci, policy, agents_by_tls, duration, cycle_seconds, train=True, record_trace=record_trace)
    finally:
        traci.close(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20, help="RL training episodes; use 200 for thesis-length runs")
    parser.add_argument("--duration", type=int, default=3600, help="SUMO simulation duration in seconds")
    parser.add_argument("--cycle-seconds", type=int, default=80, help="Control cycle length")
    parser.add_argument("--eta-values", type=parse_eta_values, default=ETAS, help="Comma-separated cyclic backpressure eta values; use one value for quick Colab smoke tests")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="chapter4_5_nine_scenario1_low_demand")
    args = parser.parse_args()

    availability = check_sumo_availability()
    if not availability.sumo:
        print(json.dumps({"status": "blocked", "errors": availability.errors}, indent=2))
        raise SystemExit(2)
    build_thesis_networks()
    import traci

    cfg = load_config("configs/chapter_4_5.json")
    set_seed(int(cfg["seed"]))
    random.seed(int(cfg["seed"]))
    config = SCENARIOS[args.scenario]
    raw_dir = Path("results/raw")
    plot_dir = Path("plots")
    episode_rows: list[dict[str, object]] = []
    queue_rows: list[dict[str, object]] = []
    eta_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    # Create one persistent agent per TLS per RL policy after briefly opening SUMO to inspect TLS/action counts.
    start_traci(traci, availability.sumo, config)
    try:
        tls_ids = list(traci.trafficlight.getIDList())
        green_counts = {tls_id: len(green_phase_indices(traci, tls_id)) for tls_id in tls_ids}
    finally:
        traci.close(False)
    policy_agents = {
        policy: {tls_id: make_agent(policy, green_counts[tls_id]) for tls_id in tls_ids}
        for policy in RL_POLICIES
    }

    for policy in RL_POLICIES:
        print(f"Starting {POLICY_LABELS[policy]} for {args.episodes} episode(s) at {args.duration}s each", flush=True)
        for episode in range(args.episodes):
            print(f"  {POLICY_LABELS[policy]} episode {episode + 1}/{args.episodes}", flush=True)
            metrics, trace = run_one_config(availability.sumo, config, policy, episode, args.duration, args.cycle_seconds, agents_by_tls=policy_agents[policy], record_trace=episode == args.episodes - 1)
            metrics.update({"scenario": args.scenario, "episode": episode, "algorithm_type": "real_sumo_rl"})
            episode_rows.append(metrics)
            for item in trace:
                item.update({"scenario": args.scenario, "episode": episode, "algorithm_type": "real_sumo_rl"})
                queue_rows.append(item)
            for agent in policy_agents[policy].values():
                if hasattr(agent, "epsilon"):
                    agent.epsilon = max(0.03, agent.epsilon * 0.96)
        tail = [row for row in episode_rows if row["policy"] == policy and row["episode"] >= args.episodes - min(10, args.episodes)]
        values = [row["mean_travel_time_seconds"] for row in tail if row.get("mean_travel_time_seconds") is not None]
        summary_rows.append({
            "scenario": args.scenario,
            "policy": policy,
            "policy_label": POLICY_LABELS[policy],
            "summary_source": f"last_{min(10, args.episodes)}_real_sumo_training_episodes",
            "mean_travel_time_seconds": None if not values else sum(values) / len(values),
        })

    best_eta = None
    best_delay = None
    best_queue = None
    best_eta_selection_metric = None
    for eta in args.eta_values:
        print(f"Starting {POLICY_LABELS[BACKPRESSURE_POLICY]} eta={eta} at {args.duration}s", flush=True)
        metrics, _trace = run_one_config(availability.sumo, config, BACKPRESSURE_POLICY, 0, args.duration, args.cycle_seconds, eta=eta)
        metrics.update({"scenario": args.scenario, "algorithm_type": "real_sumo_backpressure"})
        eta_rows.append(metrics)
        best_eta, best_delay, best_queue, best_eta_selection_metric = choose_best_eta(
            best_eta, best_delay, best_queue, metrics
        )
    if best_eta is not None:
        metrics, trace = run_one_config(availability.sumo, config, BACKPRESSURE_POLICY, 0, args.duration, args.cycle_seconds, eta=best_eta, record_trace=True)
        metrics.update({"scenario": args.scenario, "algorithm_type": "real_sumo_backpressure_best_eta"})
        summary_rows.append({
            "scenario": args.scenario,
            "policy": BACKPRESSURE_POLICY,
            "policy_label": POLICY_LABELS[BACKPRESSURE_POLICY],
            "summary_source": f"best_real_sumo_eta_by_{best_eta_selection_metric}",
            "eta": best_eta,
            "eta_selection_metric": best_eta_selection_metric,
            "mean_travel_time_seconds": metrics.get("mean_travel_time_seconds"),
        })
        for item in trace:
            item.update({"scenario": args.scenario, "episode": "best_eta", "algorithm_type": "real_sumo_backpressure_best_eta"})
            queue_rows.append(item)

    write_csv(raw_dir / "real_sumo_algorithm_episode_delay.csv", episode_rows)
    write_csv(raw_dir / "real_sumo_eta_vs_delay.csv", eta_rows)
    write_csv(raw_dir / "real_sumo_queue_vs_time.csv", queue_rows)
    write_csv(raw_dir / "real_sumo_algorithm_summary.csv", summary_rows)
    (raw_dir / "real_sumo_algorithm_comparison.json").write_text(json.dumps({
        "note": "Real SUMO/TraCI run for requested algorithms. Uses reconstructed thesis SUMO assets unless replaced with original files.",
        "scenario": args.scenario,
        "episodes": args.episodes,
        "duration_seconds": args.duration,
        "eta_values": args.eta_values,
        "best_eta": best_eta,
        "best_eta_selection_metric": best_eta_selection_metric,
        "summary": summary_rows,
    }, indent=2), encoding="utf-8")
    write_simple_svg(plot_dir / "real_sumo_algorithm_comparison.svg", "Real SUMO algorithm comparison")
    write_simple_svg(plot_dir / "real_sumo_episode_wise_delay.svg", "Real SUMO episode-wise delay")
    write_simple_svg(plot_dir / "real_sumo_eta_vs_delay.svg", "Real SUMO eta vs delay")
    write_simple_svg(plot_dir / "real_sumo_queue_vs_time.svg", "Real SUMO queue vs time")
    print(json.dumps({"summary": str(raw_dir / "real_sumo_algorithm_summary.csv"), "best_eta": best_eta}, indent=2))


if __name__ == "__main__":
    main()
