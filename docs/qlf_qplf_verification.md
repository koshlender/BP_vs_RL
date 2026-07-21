# QLF/QPLF Verification

## Corrected terminology

The thesis excerpt supplied by the user defines:

- **QLF**: Q-learning with linear function approximation.
- **QPLF**: Q-learning with piecewise-linear/action-block function approximation.

Therefore, QPLF is **not** a reward function and is **not** the backpressure pressure equation. QPLF describes how `Q_theta(x,a)` is approximated.

## Implemented QPLF agent

`src/agents/pwl_q_learning.py` implements the thesis QPLF feature construction. For two actions and four queue features `[qN, qS, qE, qW]`:

- `f'(x,a1) = [qN, qS, qE, qW, 0, 0, 0, 0]`
- `f'(x,a2) = [0, 0, 0, 0, qN, qS, qE, qW]`

Only the selected action block is active, and the update modifies only the selected action's four parameters. Features are L1-normalized to satisfy the regularity condition `sum_k |f'_k(x,a)| <= 1` for non-negative queues.

## Reward status

The supplied Chapter 4/5 excerpts do not give the exact RL reward. The current fallback QPLF experiment uses a clearly documented **queue-pressure reduction assumption** as the reward signal:

`R_t = pressure_score(previous queues) - pressure_score(current queues)`

This reward is not claimed to be the thesis reward unless the missing reward definition is supplied.

## Backpressure utilities

`src/rewards/qplf.py` contains pressure/backpressure helper functions derived from Chapter 5 equations. These helpers should be understood as **backpressure/pressure utilities**, not the QPLF approximator itself. The file name is retained for compatibility but the conceptual distinction is documented here.

## Limits

The implementation now matches the user's supplied QPLF feature-block description, but convergence claims are not asserted for the traffic experiment because the practical assumptions from the cited convergence theorem may not hold in the reconstructed/fallback environment.
