# Chapter 4-5 Reproduction Checklist

Scope is restricted to the supplied Chapter 4 and Chapter 5 excerpts.

| Item | Status | Extracted detail |
|---|---|---|
| Objective | Fully specified | Compare independent RL, semi-coordinated RL, and cyclic queue backpressure. |
| Two-intersection topology | Partially specified | Two signalized intersections; right cooperative, left non-cooperative; roads d/e 500 m, all others 3000 m; figure geometry missing. |
| Nine-intersection topology | Partially specified | Nine signalized intersections; five cooperative signals circled in figure; exact IDs inferred in config. |
| Vehicle routes | Partially specified | Two-intersection routes and intervals are listed; nine-intersection entry intervals listed but complete internal routes unavailable. |
| Scenario 1 demand | Fully specified for entry intervals | Table 4.2 first/second-half intervals captured in YAML. |
| Scenario 2 demand | Partially specified | Table 5.1 captured; thesis says "N2 100 veh", assumed 100 seconds. |
| Simulation duration | Partially specified | Nine-intersection runs are one hour; two-intersection has three hourly intervals. Short cloud runs are reduced. |
| Warm-up | Missing | No warm-up stated. |
| SUMO step length | Missing | Configurable; default 1 s. |
| Phases/actions | Partially specified | Chapter 4 green vectors and eight cooperative phase sequences implemented; Chapter 5 fallback cyclic backpressure uses four single-approach phase vectors because exact SUMO phase strings are absent. |
| Yellow duration | Partially specified | Chapter 5 says yellow is predefined after every phase and cyclic formula uses total `Y`; fallback uses Chapter 3 total `Y=16`, i.e. 4 s after each of four phases. |
| State | Partially specified | Chapter 4 independent learner uses local `Q(t)`; semi-coordinated learner uses `S(t)=[Q(t), Q'(t), a'(t)]` with documented ordering per config/test. |
| Reward | Fully specified for Chapter 4 RL | Implemented `R(t)=1/sum_i Q_i(t+1)` with explicit finite zero-queue simulation policy because the zero denominator case is not defined in the thesis text. |
| QLF | Specified in Chapter 3 | QLF retained as linear function approximation terminology, not a reward. |
| QPLF | Specified in Chapter 3 / used in Chapter 4 | QPLF is implemented as action-block piecewise-linear function approximation; backpressure pressure equations are separate Chapter 5 utilities. |
| Learning algorithm | Partially specified | RL described as independent/semi-coordinated; exact neural architecture/hyperparameters missing, so tabular Q-learning is provided as executable baseline scaffold. |
| Backpressure algorithm | Fully specified for equations | Equations for queue dynamics, weights, max-pressure diagnostic, softmax proportions, and cyclic green allocation are implemented; fallback scripts execute every cyclic phase with its assigned green time. |
| Metrics | Fully specified/extended | Cumulative stops, travel time, vehicles in network, plus requested average stops and platoon progression. |
| Reported results | Fully specified for Table 5.2 | Thesis travel times: BP 280/345, semi-RL 262/335, independent 395/397 seconds. |
