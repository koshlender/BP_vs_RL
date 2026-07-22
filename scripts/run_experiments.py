#!/usr/bin/env python
"""Run reproducible short Chapter 4/5 fallback experiments.

These experiments use the repository's deterministic queue simulator because the
original SUMO assets and TraCI are not available in the execution environment.
They are intended as executable sanity checks, not thesis-scale reproduction.
"""
from __future__ import annotations
from pathlib import Path
import csv, json, random, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agents.pwl_q_learning import PiecewiseLinearQAgent
from src.environment.simulated_env import QueueNetworkEnv
from src.rewards.backpressure import cyclic_backpressure_decision
from src.thesis.chapter_constants import THESIS_AVAILABLE_GREEN_TIME_SECONDS, THESIS_YELLOW_TIME_PER_APPROACH_SECONDS
from src.utils.config import load_config, set_seed

SCENARIOS = {
    "scenario1_low_demand": [0.12, 0.10, 0.16, 0.14],
    "scenario2_high_demand": [0.36, 0.28, 0.40, 0.34],
}


PHASES = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
IDENTITY_TURNS = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
OUTGOING_QUEUES = [0, 0, 0, 0]

def choose_backpressure_decision(state: list[float], eta: float = 0.05):
    return cyclic_backpressure_decision(
        PHASES,
        state,
        OUTGOING_QUEUES,
        IDENTITY_TURNS,
        eta=eta,
        available_green_time=THESIS_AVAILABLE_GREEN_TIME_SECONDS,
    )


def neighbouring_action_value(previous_action) -> float:
    if previous_action is None:
        return 0.0
    if isinstance(previous_action, (int, float)):
        return float(previous_action)
    if isinstance(previous_action, (list, tuple)) and previous_action:
        return float(previous_action[0])
    return 0.0

def policy_state(policy: str, queues: list[float], previous_action: int = 0) -> list[float]:
    if policy == "independent_rl_local":
        return [queues[0], queues[1], queues[2], queues[3]]
    if policy == "semi_coordinated_rl":
        return list(queues) + [neighbouring_action_value(previous_action)]
    return list(queues)


def run_episode(policy: str, demand: list[float], seed: int, agent: PiecewiseLinearQAgent | None = None, train: bool = False, duration: int = 600) -> dict:
    env = QueueNetworkEnv(duration=duration, seed=seed, demand=demand)
    queues = env.reset()
    previous_action = 0
    done = False
    total_reward = 0.0
    total_queue = 0.0
    steps = 0
    while not done:
        state = policy_state(policy, queues, previous_action)
        if policy in {"semi_coordinated_rl", "independent_rl_local"}:
            assert agent is not None
            action = agent.act(state, evaluate=not train)
        elif policy == "independent_fixed_ns":
            action = 0
        elif policy == "cyclic_queue_backpressure":
            action = None
        else:
            raise ValueError(policy)
        if policy == "cyclic_queue_backpressure":
            decision = choose_backpressure_decision(queues)
            next_queues, reward, done, info = env.step_phase_schedule(
                PHASES,
                decision.executable_green_times,
                yellow_duration_per_phase=THESIS_YELLOW_TIME_PER_APPROACH_SECONDS,
            )
        else:
            next_queues, reward, done, info = env.step(action)
        if train and agent is not None:
            next_state = policy_state(policy, next_queues, action)
            agent.update(state, action, reward, next_state, done)
        queues = next_queues
        previous_action = action if action is not None else previous_action
        total_reward += reward
        total_queue += info["queue_sum"]
        steps += 1
    metrics = env.metrics()
    metrics.update({
        "policy": policy,
        "seed": seed,
        "reward": round(total_reward, 6),
        "mean_queue": round(total_queue / max(steps, 1), 6),
        "duration_seconds": duration,
    })
    return metrics


def main() -> None:
    cfg = load_config("configs/chapter_4_5.json")
    base_seed = int(cfg["seed"])
    set_seed(base_seed)
    episodes = 30
    eval_seeds = [base_seed + 100 + i for i in range(10)]
    outdir = Path("results/raw")
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    training_rows = []
    for scenario, demand in SCENARIOS.items():
        random.seed(base_seed)
        agents = {
            "semi_coordinated_rl": PiecewiseLinearQAgent(actions=(0, 1), alpha=0.12, gamma=0.95, epsilon=0.35, state_dimension=5, use_chapter3_state_encoding=False),
            "independent_rl_local": PiecewiseLinearQAgent(actions=(0, 1), alpha=0.12, gamma=0.95, epsilon=0.35, state_dimension=4, use_chapter3_state_encoding=False),
        }
        for policy, agent in agents.items():
            for ep in range(episodes):
                row = run_episode(policy, demand, base_seed + ep, agent=agent, train=True)
                row["scenario"] = scenario
                row["episode"] = ep
                training_rows.append(row)
                agent.epsilon=max(0.03, agent.epsilon * 0.94)
        for policy in ["semi_coordinated_rl", "independent_rl_local", "independent_fixed_ns", "cyclic_queue_backpressure"]:
            eval_agent = agents.get(policy)
            eval_rows = [run_episode(policy, demand, seed, agent=eval_agent, train=False) for seed in eval_seeds]
            keys = ["average_travel_time_seconds", "total_stop_events", "observed_vehicles", "completed_vehicles", "average_stops_per_observed_vehicle", "reward", "mean_queue"]
            summary = {"scenario": scenario, "policy": policy, "eval_episodes": len(eval_rows), "platoon_progression_percentage": None, "platoon_progression_note": "undefined: deterministic fallback has aggregate queues, not vehicle IDs/corridor crossings"}
            for key in keys:
                vals = [r[key] for r in eval_rows if r[key] is not None]
                summary[f"mean_{key}"] = round(sum(vals) / len(vals), 6) if vals else None
            summary_rows.append(summary)
    with open(outdir / "experiment_training_30ep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=training_rows[0].keys())
        writer.writeheader(); writer.writerows(training_rows)
    with open(outdir / "experiment_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader(); writer.writerows(summary_rows)
    with open(outdir / "experiment_summary.json", "w") as f:
        json.dump({"note": "Deterministic queue-simulator fallback; not thesis SUMO reproduction.", "summaries": summary_rows}, f, indent=2)
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
