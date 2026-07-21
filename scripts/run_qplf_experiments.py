#!/usr/bin/env python
"""Run thesis-QPLF experiments using piecewise-linear Q-learning agents.

In the thesis excerpt, QPLF means Q-learning with piecewise-linear function
approximation: action-wise blocks f'(x,a), not a separate reward function. This
fallback script therefore labels QPLF as the Q-function approximator. Chapter 4
reinforcement-learning policies use the thesis inverse-next-queue reward.
"""
from __future__ import annotations
from pathlib import Path
import csv, json, random, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agents.pwl_q_learning import PiecewiseLinearQAgent
from src.agents.q_learning import TabularQLearningAgent
from src.environment.simulated_env import QueueNetworkEnv
from src.rewards.qplf import queue_pressure_lyapunov_function
from src.rewards.backpressure import cyclic_backpressure_decision
from src.rewards.chapter4 import chapter4_queue_reward
from src.state.state_builder import StateBuilder, StateSpec
from src.utils.config import load_config, set_seed

SCENARIOS = {
    "scenario1_low_demand": [0.12, 0.10, 0.16, 0.14],
    "scenario2_high_demand": [0.36, 0.28, 0.40, 0.34],
}
PHASES = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
IDENTITY_TURNS = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
YELLOW_SECONDS_PER_PHASE = 4

def qplf_score(queues: list[float]) -> float:
    return queue_pressure_lyapunov_function(queues, [0.0, 0.0, 0.0, 0.0])

def queue_pressure_reward(previous: list[float], current: list[float]) -> float:
    return qplf_score(previous) - qplf_score(current)

def chapter4_rl_reward(current: list[float]) -> float:
    # Thesis Chapter 4, Reward: R(t)=1/sum_i Q_i(t+1).
    return chapter4_queue_reward(current)

def backpressure_decision(state: list[float]):
    # Thesis Chapter 5 execution path: w_sigma(t) -> P_sigma(t) -> g_sigma(t).
    return cyclic_backpressure_decision(PHASES, state, [0, 0, 0, 0], IDENTITY_TURNS, eta=0.05)

def policy_state(policy: str, queues: list[float], previous_action: int) -> list[float]:
    if policy in {"full_state_independent_rl", "independent_pwl_qplf"}:
        # Chapter 4 Independent Learner: local traffic information only.
        return list(queues)
    if policy == "semi_coordinated_pwl_qplf":
        # Thesis Chapter 4, Eq. (4.1): S(t)=[Q(t),Q'(t),a'(t)].
        # In this compact fallback, q0/q1 are local incoming roads, q2/q3 are
        # neighbouring incoming roads, and previous_action is neighbouring a'(t).
        builder = StateBuilder(StateSpec(("q0", "q1"), ("q2", "q3"), ("connecting_green",)))
        return builder.build(
            {"q0": queues[0], "q1": queues[1], "q2": queues[2], "q3": queues[3]},
            {"connecting_green": previous_action if isinstance(previous_action, (int, float)) else 0},
        )
    if policy == "centralized_full_state_rl":
        # User-requested comparison baseline: one learner observes the whole compact network.
        return list(queues)
    raise ValueError(policy)

def run_episode(policy: str, demand: list[float], seed: int, agent, train: bool, duration: int = 600) -> dict:
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
            decision = backpressure_decision(queues)
            state = list(queues)
            action = None
            next_state = state
        else:
            assert agent is not None
            state = policy_state(policy, queues, previous_action)
            action = agent.act(state, evaluate=not train)
        queues, env_reward, done, info = env.step_phase_schedule(PHASES, decision.executable_green_times, YELLOW_SECONDS_PER_PHASE) if policy == "cyclic_queue_backpressure" else env.step(action)
        shaped_reward = queue_pressure_reward(prev, queues) if policy == "cyclic_queue_backpressure" else chapter4_rl_reward(queues)
        if policy != "cyclic_queue_backpressure":
            next_state = policy_state(policy, queues, action)
            if train:
                agent.update(state, action, shaped_reward, next_state, done)
        previous_action = action if action is not None else 0
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
    policies = ["full_state_independent_rl", "independent_pwl_qplf", "semi_coordinated_pwl_qplf", "centralized_full_state_rl", "cyclic_queue_backpressure"]
    training_rows = []
    summary_rows = []
    outdir = Path("results/raw")
    outdir.mkdir(parents=True, exist_ok=True)
    for scenario, demand in SCENARIOS.items():
        agents = {
            "full_state_independent_rl": TabularQLearningAgent(actions=(0, 1), alpha=0.1, epsilon=0.35),
            "independent_pwl_qplf": PiecewiseLinearQAgent(actions=(0, 1), alpha=0.05, epsilon=0.35),
            "semi_coordinated_pwl_qplf": PiecewiseLinearQAgent(actions=(0, 1), alpha=0.05, epsilon=0.35),
            "centralized_full_state_rl": TabularQLearningAgent(actions=(0, 1), alpha=0.1, epsilon=0.35),
        }
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
            summary = {"scenario": scenario, "policy": policy, "eval_episodes": len(eval_rows), "uses_qplf_piecewise_linear_approximation": "qplf" in policy, "uses_full_state_tabular_rl": "full_state" in policy, "uses_backpressure_action": policy == "cyclic_queue_backpressure", "reward_signal": "chapter4_inverse_next_queue" if policy != "cyclic_queue_backpressure" else "not_learned"}
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
        json.dump({"note": "Chapter 5 comparison includes cyclic queue backpressure, full-state tabular RL, independent QPLF, semi-coordinated QPLF, and a centralized full-state RL baseline in the compact fallback; not a full SUMO thesis reproduction.", "train_episodes": train_episodes, "summaries": summary_rows}, handle, indent=2)
    print(json.dumps(summary_rows, indent=2))

if __name__ == "__main__":
    main()
