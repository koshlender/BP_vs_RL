from __future__ import annotations

from dataclasses import dataclass

# Thesis Chapter 3, Simulation / Isolated Signal: cycle length T=80 seconds and
# total yellow time Y=16 seconds. No LaTeX source files are present in this repo;
# Chapter 4 says sum_j g_j(t)=T-Y but does not restate numeric values in the
# provided text, so Chapter 4 imports these shared thesis-cycle constants instead
# of defining divergent defaults.
THESIS_CYCLE_TIME_SECONDS = 80
THESIS_TOTAL_YELLOW_TIME_SECONDS = 16
THESIS_AVAILABLE_GREEN_TIME_SECONDS = THESIS_CYCLE_TIME_SECONDS - THESIS_TOTAL_YELLOW_TIME_SECONDS

# Thesis Chapter 3, Simulation: a,b,c are fixed as 0.0005, 0.001, 0.002.
CH3_QUEUE_LEVEL_VALUES = (0.0005, 0.001, 0.002)
CH3_GREEN_TIMES_SECONDS = (8, 16, 24, 32)

# Thesis Chapter 4, Non Cooperative Signal: green times are selected from this set.
CH4_GREEN_TIMES_SECONDS = (4, 8, 12, 16, 20, 24, 28, 32)


@dataclass(frozen=True)
class QueueCategory:
    """Symbolic Chapter 4 queue-state category.

    The thesis defines six symbolic queue categories k,l,m,n,o,p but does not
    assign independent numeric queue values to those symbols in the available
    repository/prompt sources. The numeric id is only a categorical state id for
    algorithms that require numeric feature vectors; it is not a physical queue
    length and it is not a thesis claim that k=0, l=1, ..., p=5.
    """

    label: str
    category_id: int
    lower_inclusive: int
    upper_inclusive: int | None


CH4_QUEUE_CATEGORIES = (
    QueueCategory("k", 0, 0, 10),
    QueueCategory("l", 1, 11, 20),
    QueueCategory("m", 2, 21, 30),
    QueueCategory("n", 3, 31, 40),
    QueueCategory("o", 4, 41, 50),
    QueueCategory("p", 5, 51, None),
)
CH4_QUEUE_LABEL_TO_ID = {category.label: category.category_id for category in CH4_QUEUE_CATEGORIES}
CH4_QUEUE_ID_TO_LABEL = {category.category_id: category.label for category in CH4_QUEUE_CATEGORIES}
