# Chapter 5 Backpressure and RL Comparison

The Chapter 5 implementation adds the cyclic queue backpressure equations from the supplied thesis text and retains the same Chapter 4/5 traffic scenarios already represented in the repository configuration and generated SUMO assets.

## Executed cyclic backpressure controller

The simulator execution path is:

1. observe queues at the start of the cycle;
2. compute every phase weight `w_sigma(t)`;
3. compute cyclic proportions `P_sigma(t)` with softmax;
4. compute green durations `g_sigma(t)=P_sigma(t)(T-Y)`;
5. deterministically convert fractional green durations to executable integer seconds while preserving the exact total `T-Y`;
6. execute **every** phase in order with its assigned green duration, followed by the configured yellow transition.

The compact fallback network uses four single-movement phase vectors because the thesis states that the signal framework serves only a single traffic movement during a phase.

| Phase | Permitted movements | Phase vector | Green-time source |
|---|---|---|---|
| 1 | North incoming road | `[1,0,0,0]` | `P_sigma(T-Y)` |
| 2 | South incoming road | `[0,1,0,0]` | `P_sigma(T-Y)` |
| 3 | East incoming road | `[0,0,1,0]` | `P_sigma(T-Y)` |
| 4 | West incoming road | `[0,0,0,1]` | `P_sigma(T-Y)` |

Yellow transition: the compact fallback executes `4` yellow seconds after each phase, for the thesis total yellow time `Y=16` across four phases.

## Equation traceability

| Thesis / requested component | Implementation | Status |
|---|---|---|
| Queue dynamics `Q_i(t+1)=Q_i(t)-sum_a p_ia sigma_i + sum_b p_bi sigma_b` | `src/rewards/backpressure.py::queue_dynamics` | Implemented from Chapter 5 equation |
| Phase weight `w_sigma(t)=sum_i sigma_i[Q_i(t)-sum_i' p_ii'(t)Q_ii'(t)]` | `src/rewards/backpressure.py::phase_weight` and `phase_weights` | Implemented from Chapter 5 equation |
| Max-pressure phase `argmax_sigma w_sigma(t)` | `src/rewards/backpressure.py::max_pressure_phase_index` | Reported diagnostic; does not override cyclic allocation |
| Cyclic proportion `P_sigma=exp(eta w_sigma)/sum exp(eta w_sigma')` | `src/rewards/backpressure.py::cyclic_backpressure_proportions` | Implemented from Chapter 5 equation; numerically stable softmax is mathematically equivalent |
| Cyclic green allocation `sigma(t)=P_sigma(T-Y)` | `src/rewards/backpressure.py::cyclic_backpressure_green_times` | Implemented from Chapter 5 equation |
| Executable integer durations | `src/rewards/backpressure.py::deterministic_integer_durations` | Implementation mechanism; preserves sum exactly |
| Signal execution | `src/environment/simulated_env.py::step_phase_schedule` | Executes every cyclic backpressure phase in order |
| Scenario 1 | Existing Chapter 4 nine-intersection scenario | Reused as Chapter 5 states |
| Scenario 2 | Existing high-demand scenario in config/SUMO assets | Reused as Chapter 5 states |
| Thesis comparison policies | Independent learner RL, Semi-Coordinated RL, Cyclic Queue Backpressure | Thesis reproduction policies |
| Additional comparison policies | Full-state independent RL and centralized full-state RL baseline | User-requested non-thesis baselines |

The centralized full-state RL controller is a user-requested additional baseline; it is not claimed as a thesis-reported Chapter 5 result. The eta sweep and short comparison scripts now use the same cyclic phase execution path rather than sampling a two-action proxy.
