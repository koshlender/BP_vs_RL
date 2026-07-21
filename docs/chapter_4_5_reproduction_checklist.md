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
| Phases/actions | Partially specified | Two phases assumed: north-south and east-west. Exact SUMO phase strings absent. |
| Yellow duration | Missing | Configurable; default 3 s. |
| State | Partially specified | Implemented S(t)=[Q(t), Q'(t), a'(t)] with documented ordering per config/test. |
| Reward | Missing in excerpt | Implemented configurable Lyapunov queue reduction assumption. |
| QLF | Missing in excerpt | Implemented as sum of squared queues and documented as assumption. |
| QPLF | Partially specified | Backpressure pressure equation is specified; QPLF squared positive pressure is an assumption. |
| Learning algorithm | Partially specified | RL described as independent/semi-coordinated; exact neural architecture/hyperparameters missing, so tabular Q-learning is provided as executable baseline scaffold. |
| Backpressure algorithm | Fully specified for equations | Equations 5.2-5.5 implemented for weights and cyclic green allocation. |
| Metrics | Fully specified/extended | Cumulative stops, travel time, vehicles in network, plus requested average stops and platoon progression. |
| Reported results | Fully specified for Table 5.2 | Thesis travel times: BP 280/345, semi-RL 262/335, independent 395/397 seconds. |
