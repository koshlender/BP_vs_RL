# BP_vs_RL Chapter 4-5 Reproduction

This repository recreates the Chapter 4 and Chapter 5 traffic-signal-control experiment scaffold from the supplied thesis excerpt. It includes documented assumptions, configurable demand tables, semi-coordinated state construction, action mapping, QLF/QPLF/backpressure utilities, metrics, tests, short training/evaluation scripts, raw outputs, and plots.

## Setup

```bash
python -m pip install -r requirements.txt
```

SUMO/TraCI are optional for this scaffold. The included short cloud run uses a deterministic queue simulator when exact thesis SUMO files are unavailable.

## Run

```bash
python -m pytest -q
python scripts/train.py
python scripts/evaluate.py
python scripts/plot_results.py
```

## Full reproduction status

The thesis excerpt omits exact network figures as machine-readable topology, lane connections, signal phases, Chapter-4 reward/QLF/QPLF definitions, and RL neural-network hyperparameters. These are documented in `docs/chapter_4_5_assumptions.md`; therefore generated short-run results are reproducible smoke-test outputs, not claimed to match the thesis Table 5.2 values.

## Short fallback experiment results

Run the extended fallback experiment with:

```bash
python scripts/run_experiments.py
```

This writes `results/raw/experiment_training_30ep.csv`, `results/raw/experiment_summary.csv`, and `results/raw/experiment_summary.json`. These are deterministic queue-simulator fallback results, not SUMO thesis-scale results.

## QPLF piecewise-linear fallback experiments

Run QPLF reward experiments for independent, semi-coordinated, and centralized piecewise-linear Q-learning controllers with:

```bash
python scripts/run_qplf_experiments.py
```

This writes `results/raw/qplf_training_60ep.csv`, `results/raw/qplf_summary.csv`, and `results/raw/qplf_summary.json`.

## Real SUMO/TraCI experiments

Run real headless SUMO experiments with:

```bash
python scripts/run_sumo_experiments.py
```

This command intentionally does **not** use the deterministic fallback. It requires the `sumo`, `netgenerate`, and `duarouter` binaries plus Python `traci` and `sumolib` bindings. In this cloud container the command is currently blocked because those dependencies cannot be installed, but the script is ready to run in a SUMO-enabled environment. Replace the generated grid files with the original thesis SUMO network/routes for final Chapter 4/5 reproduction.

When real SUMO/TraCI is available, `scripts/run_sumo_experiments.py` also reports a basic vehicle-ID platoon progression percentage from TraCI route/edge observations. This is a smoke-test corridor measure using the first two normal edges of each vehicle route; for thesis-quality platoon progression, configure the exact first/second intersection corridor edges from the original network.

## Thesis diagram SUMO assets

Generate the two-intersection and nine-intersection XML inputs from the thesis diagrams/tables with:

```bash
python scripts/generate_thesis_sumo.py
```

Then, in a SUMO-enabled environment, build `.net.xml` files with:

```bash
scripts/build_thesis_sumo_networks.sh
```

Generated assets are written under `sumo/thesis_ch4_ch5/`. The route files encode Table 4.1, Table 4.2 Scenario 1, and Table 5.1 Scenario 2 intervals supplied in the prompt. Geometry and turn movements are reconstructed from the supplied figures; exact lane counts/signal programs remain assumptions unless the original thesis SUMO files are provided.

## QPLF terminology correction

In this repository, QPLF should be read according to the thesis definition: **Q-learning with piecewise-linear/action-block function approximation**. It is not a separate reward function. The piecewise-linear QPLF agent is implemented in `src/agents/pwl_q_learning.py`. Any queue-pressure reward used in fallback scripts is a documented reward assumption because the supplied Chapter 4/5 excerpt does not fully specify the RL reward.
