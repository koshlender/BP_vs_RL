#!/usr/bin/env python
"""Run thesis-QPLF experiments using piecewise-linear Q-learning agents.

In the thesis excerpt, QPLF means Q-learning with piecewise-linear function
approximation: action-wise blocks f'(x,a), not a separate reward function. This
fallback script therefore labels QPLF as the Q-function approximator. The reward used
for learning remains configurable/assumed from queue-pressure reduction because the
provided excerpt does not define the exact Chapter-4 reward.
"""
from __future__ import annotations
from pathlib import Path
import csv, json, random, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agents.pwl_q_learning import PiecewiseLinearQAgent
from src.environment.simulated_env import QueueNetworkEnv
from src.rewards.qplf import backpressure_weights, queue_pressure_lyapunov_function
from src.utils.config import load_config, set_seed

SCENARIOS = {
    "scenario1_low_demand": [0.12, 0.10, 0.16, 0.14],
    "scenario2_high_demand": [0.36, 0.28, 0.40, 0.34],
}
PHASES = [[1, 1, 0, 0], [0, 0, 1, 1]]

def qplf_score(queues: list[float]) -> float:
    return queue_pressure_lyapunov_function(queues, [0.0, 0.0, 0.0, 0.0])

def queue_pressure_reward(previous: list[float], current: list[float]) -> float:
    return qplf_score(previous) - qplf_score(current)

def backpressure_action(state: list[float]) -> int:
    weights = backpressure_weights(state, [0, 0, 0, 0], PHASES)
    return 0 if weights[0] >= weights[1] else 1

def policy_state(policy: str, queues: list[float], previous_action: int) -> list[float]:
    if policy == "independent_pwl_qplf":
        # Independent learner sees only the two queues served by its current local phase family.
        return [queues[0] + queues[1], queues[2] + queues[3]]
    if policy == "semi_coordinated_pwl_qplf":
        # Semi-coordinated learner sees local queues plus a neighbouring/previous action signal.
        return list(queues) + [float(previous_action)]
    if policy == "centralized_pwl_qplf":
        # Centralized learner sees the full network queue state and previous network action.
        return list(queues) + [float(previous_action), float(sum(queues)), float(max(queues) if queues else 0.0)]
    raise ValueError(policy)

def run_episode(policy: str, demand: list[float], seed: int, agent: PiecewiseLinearQAgent | None, train: bool, duration: int = 600) -> dict:
    env = QueueNetworkEnv(duration=duration, seed=seed, demand=demand)
    queues = env.reset()
    previous_action = 0
    done = False
    total_env_reward = 0.0
    total_qplf_reward = 0.0
    total_qplf = 0.0
    total_queue = 0.0
    steps = 0
    while not done:
        prev = list(queues)
        if policy == "cyclic_queue_backpressure":
            action = backpressure_action(queues)
            state = list(queues)
            next_state = state
        else:
            assert agent is not None
            state = policy_state(policy, queues, previous_action)
            action = agent.act(state, evaluate=not train)
        queues, env_reward, done, info = env.step(action)
        shaped_reward = queue_pressure_reward(prev, queues)
        if policy != "cyclic_queue_backpressure":
            next_state = policy_state(policy, queues, action)
            if train:
                agent.update(state, action, shaped_reward, next_state, done)
        previous_action = action
        total_env_reward += env_reward
        total_qplf_reward += shaped_reward
        total_qplf += qplf_score(queues)
        total_queue += info["queue_sum"]
        steps += 1
    metrics = env.metrics()
    metrics.update({
        "policy": policy,
        "seed": seed,
        "env_reward": round(total_env_reward, 6),
        "qplf_reward": round(total_qplf_reward, 6),
        "mean_qplf": round(total_qplf / max(steps, 1), 6),
        "mean_queue": round(total_queue / max(steps, 1), 6),
        "mean_delay_seconds": round(total_queue / max(metrics.get("completed_vehicles", 0), 1), 6),
        "travel_time_proxy_seconds": round(120.0 + total_queue / max(metrics.get("completed_vehicles", 0), 1), 6),
        "duration_seconds": duration,
    })
    return metrics

def main() -> None:
    cfg = load_config("configs/chapter_4_5.json")
    base_seed = int(cfg["seed"])
    set_seed(base_seed)
    train_episodes = 60
    eval_seeds = [base_seed + 200 + i for i in range(10)]
    policies = ["independent_pwl_qplf", "semi_coordinated_pwl_qplf", "centralized_pwl_qplf", "cyclic_queue_backpressure"]
    training_rows = []
    summary_rows = []
    outdir = Path("results/raw")
    outdir.mkdir(parents=True, exist_ok=True)
    for scenario, demand in SCENARIOS.items():
        agents = {policy: PiecewiseLinearQAgent(alpha=0.05, epsilon=0.35) for policy in policies if policy != "cyclic_queue_backpressure"}
        for policy, agent in agents.items():
            random.seed(base_seed)
            for episode in range(train_episodes):
                row = run_episode(policy, demand, base_seed + episode, agent, train=True)
                row.update({"scenario": scenario, "episode": episode})
                training_rows.append(row)
                agent.epsilon = max(0.03, agent.epsilon * 0.96)
        for policy in policies:
            agent = agents.get(policy)
            eval_rows = [run_episode(policy, demand, seed, agent, train=False) for seed in eval_seeds]
            summary = {"scenario": scenario, "policy": policy, "eval_episodes": len(eval_rows), "uses_qplf_piecewise_linear_approximation": policy != "cyclic_queue_backpressure", "uses_backpressure_action": policy == "cyclic_queue_backpressure", "reward_signal": "queue_pressure_reduction_assumption" if policy != "cyclic_queue_backpressure" else "not_learned"}
            for key in ["travel_time_proxy_seconds", "mean_delay_seconds", "average_travel_time_seconds", "completed_vehicles", "total_stop_events", "average_stops_per_observed_vehicle", "env_reward", "qplf_reward", "mean_qplf", "mean_queue"]:
                vals = [row[key] for row in eval_rows if row[key] is not None]
                summary[f"mean_{key}"] = round(sum(vals) / len(vals), 6) if vals else None
            summary_rows.append(summary)
    with open(outdir / "qplf_training_60ep.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=training_rows[0].keys())
        writer.writeheader(); writer.writerows(training_rows)
    with open(outdir / "qplf_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader(); writer.writerows(summary_rows)
    with open(outdir / "qplf_summary.json", "w") as handle:
        json.dump({"note": "QPLF here means thesis piecewise-linear Q-function approximation; reward is a documented queue-pressure reduction assumption; not SUMO thesis reproduction.", "train_episodes": train_episodes, "summaries": summary_rows}, handle, indent=2)
    print(json.dumps(summary_rows, indent=2))

if __name__ == "__main__":
    main()
