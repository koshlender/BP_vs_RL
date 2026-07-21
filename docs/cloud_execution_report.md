# Cloud Execution Report

This report is generated for the Codex cloud run and updated after tests.

## Environment commands
- `cat /etc/os-release`
- `python --version`
- `nproc && free -h && (command -v nvidia-smi && nvidia-smi || true)`
- `command -v sumo && sumo --version || true`
- `python -c "import traci, sumolib"`

## Executed experiment commands
- `python -m pytest -q`
- `python scripts/train.py`
- `python scripts/evaluate.py`
- `python scripts/plot_results.py`

## Status
Environment observed: Ubuntu 24.04.4 LTS, Python 3.12.13, 3 CPU cores, 17 GiB memory, no detected GPU output, SUMO binary not found, and `traci`/`sumolib` unavailable. Network package installation was blocked by HTTP 403, so tests use the standard-library deterministic simulator. Final status: unit/integration tests passed; short training completed; short evaluation completed; plot artifacts generated.

## Additional experiment run requested after initial scaffold

Command: `python scripts/run_experiments.py`

The run completed 30 training episodes per scenario for the fallback semi-coordinated RL controller and 10 deterministic evaluation episodes per policy/scenario. Raw outputs were written to:

- `results/raw/experiment_training_30ep.csv`
- `results/raw/experiment_summary.csv`
- `results/raw/experiment_summary.json`

Important limitation: these are deterministic queue-simulator fallback results because SUMO/TraCI and original thesis network files are unavailable; they are not claimed to reproduce thesis Table 5.2 numerically.

## Platoon progression ratio follow-up

Command: `python scripts/run_experiments.py`

Result: platoon progression percentage is undefined for every fallback case because the deterministic fallback simulator stores aggregate queue counts only. The thesis/requested formula requires unique vehicle IDs and first/second-intersection corridor crossing events, which are TraCI/SUMO trajectory data not available in this fallback run. The undefined results are written to `results/raw/platoon_progression_summary.csv` and `.json`.

## QPLF piecewise-linear Q-learning and centralized follow-up

Command: `python scripts/run_qplf_experiments.py`

Result: completed 60 fallback training episodes per RL policy/scenario and 10 deterministic evaluation episodes for independent QPLF piecewise-linear Q-learning, semi-coordinated QPLF piecewise-linear Q-learning, centralized QPLF piecewise-linear Q-learning, and cyclic queue backpressure. Raw outputs were written to `results/raw/qplf_training_60ep.csv`, `results/raw/qplf_summary.csv`, and `results/raw/qplf_summary.json`.

## QPLF delay metric correction

Observation: the previous fallback `average_travel_time_seconds` came from the toy environment's final-queue proxy, so several controllers had identical means even when their cumulative queues differed. Correction: `scripts/run_qplf_experiments.py` now also reports `mean_delay_seconds = cumulative_queue_area / completed_vehicles` and `travel_time_proxy_seconds = 120 + mean_delay_seconds`, which are based on the whole evaluation episode rather than only the final queue snapshot.

## Real SUMO experiment attempt

Command: `python scripts/run_sumo_experiments.py`

Status: blocked in this container. The command intentionally avoids the deterministic fallback and requires real SUMO/TraCI dependencies. The observed missing components were: `sumo`, `netgenerate`, Python `traci`, and Python `sumolib`. An attempt to install `sumo sumo-tools` with `apt-get` was blocked by HTTP 403 responses from the Ubuntu package repositories.

## Independent learner RL fallback correction

Command: `python scripts/run_experiments.py`

Result: the fallback experiment now includes `independent_rl_local`, which is a tabular RL controller trained with only its own local queue state and no neighbouring action/state field. This replaces using only `independent_fixed_ns` when comparing learned controllers. The fixed controller is retained only as a starvation sanity baseline.

## Real SUMO platoon progression support

The real SUMO runner now includes a TraCI vehicle-ID platoon progression tracker. In a SUMO-enabled environment such as Google Colab, `python scripts/run_sumo_experiments.py` will output `platoon_eligible_vehicles`, `platoon_progressed_vehicles`, and `platoon_progression_percentage`. This cannot be computed in the current container because SUMO/TraCI are missing.
