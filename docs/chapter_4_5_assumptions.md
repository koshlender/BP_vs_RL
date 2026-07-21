# Chapter 4-5 Assumptions and Missing Information

## Network geometry
- Missing: Figures 4.1 and 4.2 geometry, node coordinates, lane counts, turn connections, and signal programs.
- Assumption: Placeholder SUMO XML and a deterministic queue simulator are included so algorithms and metrics can run in headless CI.
- Reasonable because: The excerpt gives demand tables and controller equations but not enough to reconstruct exact SUMO networks.
- Effect: Numerical results are smoke-test reproductions, not validated thesis-scale replication.
- Verification: Obtain original `.net.xml`, `.rou.xml`, and thesis figures.

## Scenario 2 N2 demand
- Missing: Table 5.1 says `N2 100 veh` rather than seconds.
- Assumption: Treat as 100-second generation interval.
- Effect: Demand at N2 may be under/over-estimated.
- Verification: Check original thesis/source files.

## RL method details
- Missing: Chapter excerpt lacks neural architecture, reward equation, replay settings, and exploration schedule.
- Assumption: A tabular Q-learning scaffold with configurable hyperparameters is used for executable short runs.
- Effect: Short-run RL results must not be compared as faithful 200-episode reproduction.
- Verification: Obtain Chapter 3 or original code if it defines QLF/QPLF and RL hyperparameters.

## QLF/QPLF
- Missing: Exact Chapter-4 QLF/QPLF definitions are not present in the supplied text.
- Assumption: QLF is sum of squared queues; QPLF is squared positive pressure based on Equation 5.2.
- Effect: Mathematical verification is limited to these explicit assumptions.
- Verification: Compare against the thesis sections where QLF/QPLF are originally defined.
