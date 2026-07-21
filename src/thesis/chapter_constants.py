from __future__ import annotations

from dataclasses import dataclass

# Chapter 3 supplies the numeric cycle/yellow values used by Chapter 4's
# equation sum_j g_j(t)=T-Y; Chapter 4 itself changes the state bins, action
# candidates, and reward definition.
THESIS_CYCLE_TIME_SECONDS = 80
THESIS_TOTAL_YELLOW_TIME_SECONDS = 16
THESIS_YELLOW_TIME_PER_APPROACH_SECONDS = 4
THESIS_AVAILABLE_GREEN_TIME_SECONDS = THESIS_CYCLE_TIME_SECONDS - THESIS_TOTAL_YELLOW_TIME_SECONDS

# Chapter 3 isolated-signal constants retained for the Chapter 3 agents.
CH3_QUEUE_LEVEL_VALUES = (0.0005, 0.001, 0.002)
CH3_QUEUE_LEVEL_LABELS = ("a", "b", "c")
CH3_QUEUE_THRESHOLDS = (25, 50)
CH3_GREEN_TIMES_SECONDS = (8, 16, 24, 32)

# Chapter 4 non-cooperative/cooperative signal action candidates.
CH4_GREEN_TIMES_SECONDS = (4, 8, 12, 16, 20, 24, 28, 32)


@dataclass(frozen=True)
class QueueCategory:
    """Symbolic Chapter 4 queue-state category k,l,m,n,o,p."""

    label: str
    category_id: int
    lower_inclusive: int
    upper_inclusive: int | None


# Chapter 4 writes strict lower bounds such as 11<q'<=20, which leaves integer
# boundary gaps at 11,21,31,41. For integer queue lengths we use the consecutive
# partition implied by the text's 10-vehicle buckets.
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
