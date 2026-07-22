#!/usr/bin/env python
"""Run Chapter 4/5 thesis-style algorithm comparisons and plots.

This script produces one place for the requested comparison artifacts:

* episode-wise delay for RL algorithms (independent/semi-coordinated x full RL/QPLF),
* eta-vs-delay for cyclic queue backpressure with eta from 0.1 to 1.2,
* final comparative delay table/graph, and
* queue-vs-time traces for the last RL training episode and best-eta backpressure.

The implementation uses the repository's compact queue simulator and thesis-equation
control APIs, so it is reproducible in Colab/CI without TraCI. Its delay values
are queue-delay surrogates, not SUMO vehicle trip times. Real SUMO static-network
metrics remain in ``scripts/run_thesis_sumo_experiments.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.pwl_q_learning import PiecewiseLinearQAgent
from src.environment.simulated_env import QueueNetworkEnv
from src.rewards.backpressure import cyclic_backpressure_decision
from src.rewards.chapter4 import chapter4_queue_reward
from src.state.state_builder import chapter4_quantize_queue
from src.thesis.chapter_constants import THESIS_AVAILABLE_GREEN_TIME_SECONDS, THESIS_YELLOW_TIME_PER_APPROACH_SECONDS
from src.utils.config import load_config, set_seed

SCENARIOS = {
    "scenario1_low_demand": [0.12, 0.10, 0.16, 0.14],
    "scenario2_high_demand": [0.36, 0.28, 0.40, 0.34],
}
RL_POLICIES = [
    "independent_full_rl",
    "independent_qplf",
    "semi_coordinated_full_rl",
    "semi_coordinated_qplf",
]
BACKPRESSURE_POLICY = "cyclic_queue_backpressure"
ETAS = [round(i / 10, 1) for i in range(1, 13)]
PHASES = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
IDENTITY_TURNS = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
OUTGOING_QUEUES = [0, 0, 0, 0]

POLICY_LABELS = {
    "independent_full_rl": "Independent Learner - Full RL",
    "independent_qplf": "Independent Learner - QPLF",
    "semi_coordinated_full_rl": "Semi-Coordinated - Full RL",
    "semi_coordinated_qplf": "Semi-Coordinated - QPLF",
    BACKPRESSURE_POLICY: "Cyclic Queue Backpressure",
}


class FlexibleTabularAgent:
    """Small tabular full-state Q learner for compact Chapter 4 comparison states."""

    def __init__(self, actions=(0, 1), alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.35):
        self.actions = tuple(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q: dict[tuple[float, ...], list[float]] = {}

    def _key(self, state: list[float]) -> tuple[float, ...]:
        # Full-state RL is tabular, so use the Chapter 4 queue buckets instead
        # of raw continuous queue values; otherwise the tabular learner sees
        # almost every queue vector as a new state.
        key: list[float] = []
        for idx, value in enumerate(state):
            key.append(float(chapter4_quantize_queue(value)) if idx < 4 else float(value))
        return tuple(key)

    def values(self, state: list[float]) -> list[float]:
        key = self._key(state)
        self.q.setdefault(key, [0.0 for _ in self.actions])
        return self.q[key]

    def act(self, state: list[float], evaluate: bool = False):
        if (not evaluate) and random.random() < self.epsilon:
            return random.choice(self.actions)
        vals = self.values(state)
        return self.actions[max(range(len(vals)), key=lambda idx: vals[idx])]

    def update(self, state, action, reward, next_state, done) -> None:
        vals = self.values(state)
        idx = self.actions.index(action)
        target = reward + (0.0 if done else self.gamma * max(self.values(next_state)))
        vals[idx] += self.alpha * (target - vals[idx])


@dataclass
class EpisodeResult:
    metrics: dict[str, object]
    trace: list[dict[str, object]]


def make_agent(policy: str):
    if policy == "independent_full_rl":
        return FlexibleTabularAgent(actions=(0, 1), alpha=0.1, epsilon=0.35)
    if policy == "semi_coordinated_full_rl":
        return FlexibleTabularAgent(actions=(0, 1), alpha=0.1, epsilon=0.35)
    if policy == "independent_qplf":
        return PiecewiseLinearQAgent(actions=(0, 1), alpha=0.05, epsilon=0.35, state_dimension=4, use_chapter3_state_encoding=False)
    if policy == "semi_coordinated_qplf":
        return PiecewiseLinearQAgent(actions=(0, 1), alpha=0.05, epsilon=0.35, state_dimension=5, use_chapter3_state_encoding=False)
    raise ValueError(policy)


def rl_state(policy: str, queues: list[float], previous_action: int | float) -> list[float]:
    if policy in {"independent_full_rl", "independent_qplf"}:
        return list(queues)
    if policy in {"semi_coordinated_full_rl", "semi_coordinated_qplf"}:
        # Compact cooperative state: local queues plus neighboring/previous connecting green proxy.
        return list(queues) + [float(previous_action)]
    raise ValueError(policy)


def mean_delay_from_queue(total_queue: float, completed: int) -> float:
    return total_queue / max(completed, 1)


def run_rl_episode(policy: str, demand: list[float], seed: int, agent, train: bool, duration: int, record_trace: bool = False) -> EpisodeResult:
    env = QueueNetworkEnv(duration=duration, seed=seed, demand=demand)
    queues = env.reset()
    previous_action: int | float = 0
    total_queue = 0.0
    step_count = 0
    done = False
    trace: list[dict[str, object]] = []
    while not done:
        state = rl_state(policy, queues, previous_action)
        action = agent.act(state, evaluate=not train)
        queues, env_reward, done, info = env.step(action)
        reward = chapter4_queue_reward(queues)
        next_state = rl_state(policy, queues, action)
        if train:
            agent.update(state, action, reward, next_state, done)
        previous_action = action
        total_queue += info["queue_sum"]
        step_count += 1
        if record_trace:
            trace.append({
                "time_seconds": env.t,
                "queue_north": queues[0],
                "queue_south": queues[1],
                "queue_east": queues[2],
                "queue_west": queues[3],
                "total_queue": sum(queues),
                "action": action,
                "reward": reward,
                "env_reward": env_reward,
            })
    metrics = env.metrics()
    delay = mean_delay_from_queue(total_queue, metrics.get("completed_vehicles", 0))
    metrics.update({
        "policy": policy,
        "algorithm": POLICY_LABELS[policy],
        "seed": seed,
        "mean_delay_seconds": round(delay, 6),
        "queue_delay_surrogate_seconds": round(delay, 6),
        "mean_queue": round(total_queue / max(step_count, 1), 6),
        "duration_seconds": duration,
    })
    return EpisodeResult(metrics=metrics, trace=trace)


def run_backpressure_episode(demand: list[float], seed: int, eta: float, duration: int, record_trace: bool = False) -> EpisodeResult:
    env = QueueNetworkEnv(duration=duration, seed=seed, demand=demand)
    queues = env.reset()
    total_queue_area = 0.0
    samples = 0
    cycles = 0
    trace: list[dict[str, object]] = []
    while env.t < duration:
        decision = cyclic_backpressure_decision(
            PHASES,
            queues,
            OUTGOING_QUEUES,
            IDENTITY_TURNS,
            eta=eta,
            available_green_time=THESIS_AVAILABLE_GREEN_TIME_SECONDS,
        )
        cycles += 1
        green_times = list(decision.executable_green_times)
        for phase, green_seconds in zip(PHASES, green_times):
            for _ in range(int(green_seconds)):
                if env.t >= duration:
                    break
                env._arrival_step()
                departed = 0
                for idx, sigma_i in enumerate(phase):
                    if float(sigma_i) > 0:
                        served = min(env.queues[idx], float(sigma_i))
                        env.queues[idx] -= served
                        departed += int(served)
                env.completed += departed
                env._record_stops()
                env.t += 1
                qsum = sum(env.queues)
                total_queue_area += qsum
                samples += 1
                if record_trace:
                    trace.append({
                        "time_seconds": env.t,
                        "queue_north": env.queues[0],
                        "queue_south": env.queues[1],
                        "queue_east": env.queues[2],
                        "queue_west": env.queues[3],
                        "total_queue": qsum,
                        "eta": eta,
                        "green_north": green_times[0],
                        "green_south": green_times[1],
                        "green_east": green_times[2],
                        "green_west": green_times[3],
                    })
            for _ in range(int(THESIS_YELLOW_TIME_PER_APPROACH_SECONDS)):
                if env.t >= duration:
                    break
                env._arrival_step()
                env._record_stops()
                env.t += 1
                qsum = sum(env.queues)
                total_queue_area += qsum
                samples += 1
                if record_trace:
                    trace.append({
                        "time_seconds": env.t,
                        "queue_north": env.queues[0],
                        "queue_south": env.queues[1],
                        "queue_east": env.queues[2],
                        "queue_west": env.queues[3],
                        "total_queue": qsum,
                        "eta": eta,
                        "green_north": green_times[0],
                        "green_south": green_times[1],
                        "green_east": green_times[2],
                        "green_west": green_times[3],
                    })
        queues = list(env.queues)
    metrics = env.metrics()
    delay = mean_delay_from_queue(total_queue_area, metrics.get("completed_vehicles", 0))
    metrics.update({
        "policy": BACKPRESSURE_POLICY,
        "algorithm": POLICY_LABELS[BACKPRESSURE_POLICY],
        "seed": seed,
        "eta": eta,
        "mean_delay_seconds": round(delay, 6),
        "queue_delay_surrogate_seconds": round(delay, 6),
        "mean_queue": round(total_queue_area / max(samples, 1), 6),
        "cycles": cycles,
        "duration_seconds": duration,
    })
    return EpisodeResult(metrics=metrics, trace=trace)


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


def summarize_rows(rows: Iterable[dict[str, object]], group_keys: tuple[str, ...], value_keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    summaries: list[dict[str, object]] = []
    for group, items in sorted(groups.items(), key=lambda kv: kv[0]):
        out = {key: value for key, value in zip(group_keys, group)}
        out["episodes"] = len(items)
        for key in value_keys:
            vals = [float(item[key]) for item in items if item.get(key) is not None]
            out[f"mean_{key}"] = round(sum(vals) / len(vals), 6) if vals else None
        summaries.append(out)
    return summaries


def svg_line_chart(path: Path, title: str, rows: list[dict[str, object]], x_key: str, y_key: str, series_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 560
    left, right, top, bottom = 80, 40, 70, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [float(r[x_key]) for r in rows]
    ys = [float(r[y_key]) for r in rows]
    if not xs or not ys:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
        return
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax: xmax = xmin + 1
    if ymin == ymax: ymax = ymin + 1
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    series = sorted({str(r[series_key]) for r in rows})
    def px(x: float) -> float: return left + (x - xmin) / (xmax - xmin) * plot_w
    def py(y: float) -> float: return top + plot_h - (y - ymin) / (ymax - ymin) * plot_h
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{left}" y="35" font-family="sans-serif" font-size="24">{title}</text>', f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black"/>']
    lines.append(f'<text x="{left+plot_w/2-40}" y="{height-25}" font-family="sans-serif" font-size="14">{x_key}</text>')
    lines.append(f'<text x="15" y="{top+plot_h/2}" font-family="sans-serif" font-size="14" transform="rotate(-90 15,{top+plot_h/2})">{y_key}</text>')
    for idx, name in enumerate(series):
        pts = sorted((float(r[x_key]), float(r[y_key])) for r in rows if str(r[series_key]) == name)
        point_str = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in pts)
        color = colors[idx % len(colors)]
        lines.append(f'<polyline points="{point_str}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in pts:
            lines.append(f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="2.5" fill="{color}"/>')
        legend_y = top + 20 * idx
        lines.append(f'<rect x="{left+plot_w-230}" y="{legend_y-10}" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{left+plot_w-212}" y="{legend_y}" font-family="sans-serif" font-size="12">{name}</text>')
    lines.append(f'<text x="{left}" y="{top+plot_h+20}" font-family="sans-serif" font-size="12">x: {xmin:g} to {xmax:g}; y: {ymin:.3f} to {ymax:.3f}</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def svg_bar_chart(path: Path, title: str, rows: list[dict[str, object]], label_key: str, value_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 980, 560
    left, top, bottom = 90, 70, 150
    plot_h = height - top - bottom
    vals = [float(r[value_key]) for r in rows]
    vmax = max(vals) if vals else 1.0
    bar_w = max(20, (width - left - 40) / max(len(rows), 1) * 0.7)
    gap = max(10, (width - left - 40) / max(len(rows), 1) * 0.3)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{left}" y="35" font-family="sans-serif" font-size="24">{title}</text>', f'<line x1="{left}" y1="{top+plot_h}" x2="{width-30}" y2="{top+plot_h}" stroke="black"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black"/>']
    for idx, row in enumerate(rows):
        val = float(row[value_key])
        x = left + 15 + idx * (bar_w + gap)
        h = 0 if vmax == 0 else val / vmax * plot_h
        y = top + plot_h - h
        label = str(row[label_key])
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#1f77b4"/>')
        lines.append(f'<text x="{x:.1f}" y="{y-5:.1f}" font-family="sans-serif" font-size="11">{val:.2f}</text>')
        lines.append(f'<text x="{x+bar_w/2:.1f}" y="{top+plot_h+18}" font-family="sans-serif" font-size="11" text-anchor="end" transform="rotate(-35 {x+bar_w/2:.1f},{top+plot_h+18})">{label}</text>')
    lines.append(f'<text x="15" y="{top+plot_h/2}" font-family="sans-serif" font-size="14" transform="rotate(-90 15,{top+plot_h/2})">{value_key}</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=60, help="RL training episodes; use 200 for thesis-length runs")
    parser.add_argument("--duration", type=int, default=600, help="Episode duration in seconds")
    args = parser.parse_args()

    cfg = load_config("configs/chapter_4_5.json")
    base_seed = int(cfg["seed"])
    set_seed(base_seed)
    random.seed(base_seed)

    raw_dir = Path("results/raw")
    plot_dir = Path("plots")
    raw_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    episode_rows: list[dict[str, object]] = []
    queue_rows: list[dict[str, object]] = []
    eta_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    # RL episode-wise delay and final-episode queue traces.
    final_agents: dict[tuple[str, str], object] = {}
    for scenario, demand in SCENARIOS.items():
        for policy in RL_POLICIES:
            agent = make_agent(policy)
            final_agents[(scenario, policy)] = agent
            for episode in range(args.episodes):
                result = run_rl_episode(policy, demand, base_seed + episode, agent, train=True, duration=args.duration, record_trace=episode == args.episodes - 1)
                row = dict(result.metrics)
                row.update({"scenario": scenario, "episode": episode, "policy_label": POLICY_LABELS[policy]})
                episode_rows.append(row)
                if episode == args.episodes - 1:
                    for trace_row in result.trace:
                        trace_row.update({"scenario": scenario, "episode": episode, "policy": policy, "policy_label": POLICY_LABELS[policy], "algorithm_type": "rl_last_episode"})
                        queue_rows.append(trace_row)
                if hasattr(agent, "epsilon"):
                    agent.epsilon = max(0.03, agent.epsilon * 0.96)

    # Eta-vs-delay for cyclic queue backpressure, eta = 0.1 ... 1.2.
    best_eta_by_scenario: dict[str, float] = {}
    for scenario, demand in SCENARIOS.items():
        scenario_eta_rows: list[dict[str, object]] = []
        for eta in ETAS:
            result = run_backpressure_episode(demand, base_seed + 999, eta, duration=args.duration)
            row = dict(result.metrics)
            row.update({"scenario": scenario, "policy_label": POLICY_LABELS[BACKPRESSURE_POLICY]})
            eta_rows.append(row)
            scenario_eta_rows.append(row)
        best = min(scenario_eta_rows, key=lambda r: float(r["mean_delay_seconds"]))
        best_eta_by_scenario[scenario] = float(best["eta"])
        trace_result = run_backpressure_episode(demand, base_seed + 1001, float(best["eta"]), duration=args.duration, record_trace=True)
        for trace_row in trace_result.trace:
            trace_row.update({"scenario": scenario, "episode": "best_eta", "policy": BACKPRESSURE_POLICY, "policy_label": POLICY_LABELS[BACKPRESSURE_POLICY], "algorithm_type": "backpressure_best_eta"})
            queue_rows.append(trace_row)

    # Final comparison: mean over the last 10 RL episodes plus best-eta backpressure.
    tail_window = min(10, args.episodes)
    tail_rows = [r for r in episode_rows if int(r["episode"]) >= args.episodes - tail_window]
    rl_summary = summarize_rows(tail_rows, ("scenario", "policy", "policy_label"), ("mean_delay_seconds", "queue_delay_surrogate_seconds", "mean_queue", "total_stop_events", "completed_vehicles"))
    for row in rl_summary:
        row["summary_source"] = f"last_{tail_window}_training_episodes"
        summary_rows.append(row)
    for scenario in SCENARIOS:
        best_eta = best_eta_by_scenario[scenario]
        bp_row = next(r for r in eta_rows if r["scenario"] == scenario and float(r["eta"]) == best_eta)
        summary_rows.append({
            "scenario": scenario,
            "policy": BACKPRESSURE_POLICY,
            "policy_label": POLICY_LABELS[BACKPRESSURE_POLICY],
            "episodes": 1,
            "mean_mean_delay_seconds": bp_row["mean_delay_seconds"],
            "mean_queue_delay_surrogate_seconds": bp_row["queue_delay_surrogate_seconds"],
            "mean_mean_queue": bp_row["mean_queue"],
            "mean_total_stop_events": bp_row["total_stop_events"],
            "mean_completed_vehicles": bp_row["completed_vehicles"],
            "eta": best_eta,
            "summary_source": "best_eta_backpressure",
        })

    write_csv(raw_dir / "thesis_algorithm_episode_delay.csv", episode_rows)
    write_csv(raw_dir / "thesis_eta_vs_delay.csv", eta_rows)
    write_csv(raw_dir / "thesis_queue_vs_time.csv", queue_rows)
    write_csv(raw_dir / "thesis_algorithm_summary.csv", summary_rows)
    (raw_dir / "thesis_algorithm_comparison.json").write_text(json.dumps({
        "note": "Thesis-style comparison using the compact queue simulator: RL episode-wise queue-delay surrogate, eta sweep 0.1..1.2, and queue traces for last RL episode / best-eta backpressure. These are not SUMO trip-time measurements.",
        "episodes": args.episodes,
        "duration_seconds": args.duration,
        "etas": ETAS,
        "best_eta_by_scenario": best_eta_by_scenario,
        "summary": summary_rows,
    }, indent=2), encoding="utf-8")

    svg_line_chart(plot_dir / "thesis_episode_wise_delay.svg", "Episode-wise delay for RL algorithms", episode_rows, "episode", "mean_delay_seconds", "policy_label")
    svg_line_chart(plot_dir / "thesis_eta_vs_delay.svg", "Eta vs delay for cyclic queue backpressure", eta_rows, "eta", "mean_delay_seconds", "scenario")
    svg_line_chart(plot_dir / "thesis_queue_vs_time.svg", "Queue vs time: last RL episode and best eta backpressure", queue_rows, "time_seconds", "total_queue", "policy_label")
    svg_bar_chart(plot_dir / "thesis_algorithm_comparison.svg", "Comparative mean delay by algorithm", summary_rows, "policy_label", "mean_mean_delay_seconds")

    print(json.dumps({
        "raw_outputs": [
            str(raw_dir / "thesis_algorithm_episode_delay.csv"),
            str(raw_dir / "thesis_eta_vs_delay.csv"),
            str(raw_dir / "thesis_queue_vs_time.csv"),
            str(raw_dir / "thesis_algorithm_summary.csv"),
            str(raw_dir / "thesis_algorithm_comparison.json"),
        ],
        "plots": [
            str(plot_dir / "thesis_episode_wise_delay.svg"),
            str(plot_dir / "thesis_eta_vs_delay.svg"),
            str(plot_dir / "thesis_queue_vs_time.svg"),
            str(plot_dir / "thesis_algorithm_comparison.svg"),
        ],
        "best_eta_by_scenario": best_eta_by_scenario,
    }, indent=2))


if __name__ == "__main__":
    main()
