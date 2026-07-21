# QLF/QPLF Verification

## Implemented QLF
`QLF(Q)=sum_i Q_i^2`, implemented in `src/rewards/qlf.py`.

Properties checked numerically:
- Zero queues return 0.
- Equal queues produce symmetric values.
- Larger queues increase the function monotonically for non-negative queues.
- Negative queues are rejected.

Reward assumption: `reward = QLF(previous) - QLF(current)`, so decreasing congestion is positive.

## Implemented QPLF
For incoming queues `Q_i`, downstream queues `Q_j`, and turn fractions `p_ij`, pressure follows Chapter 5 Equation 5.2:
`pressure_i = Q_i - sum_j p_ij Q_j`.

The implemented Lyapunov-like function is:
`QPLF = sum_i max(pressure_i, 0)^2`.

## Cyclic backpressure
The implementation maps Equation 5.5 to a numerically stable softmax:
`P_sigma = exp(eta w_sigma) / sum_sigma' exp(eta w_sigma')`, and Equation 5.4 to green times summing to `T-Y`.

## Limits
The supplied Chapter 4/5 excerpt does not contain the original QLF/QPLF proof. Therefore this repository verifies numerical behavior and equation-to-code consistency only; it does not claim convergence.

## QPLF piecewise-linear feature correction

The `PiecewiseLinearQAgent` now follows the thesis block-feature definition for QPLF: with two actions and state queues `[qN, qS, qE, qW]`, action `a1` activates the first four components and zeros the second four, while action `a2` zeros the first four and activates the second four. Features are normalized so the regularity condition `sum_k |f'_k(x,a)| <= 1` holds for non-negative queues. This replaces the earlier hinge-feature approximation, which was a generic piecewise-linear approximator but not the QPLF block construction described in the thesis excerpt.
