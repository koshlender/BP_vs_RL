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
- Specified by Chapter 4: the Independent Learner System uses only local traffic information, while the Semi-Coordinated System adds neighboring non-cooperative queues and actions.
- Specified by Chapter 4: unless otherwise stated, agents use QPLF from Chapter 3 as the function approximator.
- Remaining limitation: exact original learning rates/episode counts for every reported figure are not fully machine-readable in this repository.

## QLF/QPLF
- Chapter 4 uses QPLF as the underlying reinforcement-learning approximator; QPLF is not a reward function.
- Implementation: independent QPLF receives local `Q(t)` only; semi-coordinated QPLF receives `S(t)=[Q(t),Q'(t),a'(t)]`.
- Chapter 5 pressure helpers remain separate backpressure utilities, not the QPLF approximator.

## Reconstructed SUMO network assets from supplied figures
- Missing: Exact SUMO node coordinates, lane counts, connections, and signal phase programs from the original thesis implementation.
- Assumption: `scripts/generate_thesis_sumo.py` reconstructs the two-intersection and nine-intersection layouts from the supplied figures using one lane per directed road, signalized internal intersections, 500 m two-signal connector roads `d/e`, and the table-provided traffic generation intervals.
- Reasonable because: The figures identify road/entry labels and tables specify demand intervals, but they do not provide machine-readable SUMO files.
- Effect: The generated SUMO files should match the high-level topology and demand tables, but microscopic results may differ from the thesis if original lane counts, turning priorities, or signal phases differ.
- Verification: Compare generated `.nod.xml`, `.edg.xml`, `.rou.xml`, and `netconvert` output against the original thesis SUMO files if they become available.

## QPLF terminology correction
- Correction: QPLF is treated as Q-learning with piecewise-linear/action-block function approximation, not as a reward function.
- Chapter 4 reward is `R(t)=1/sum_i Q_i(t+1)`.
- Implementation: Fallback learning scripts use the Chapter 4 inverse-next-queue reward and make the undefined zero-queue case explicit with a finite default safeguard.
- Effect: QPLF experiment outputs test the Chapter 4 semi-coordinated/independent learner structure; numerical values may still change with original SUMO files.
