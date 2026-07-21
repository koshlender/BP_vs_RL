from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class PhasePlan:
    green_phases: dict[int, int]
    yellow_phases: dict[tuple[int,int], int]
    min_green: int = 10
    yellow_duration: int = 3

def action_to_phase(action: int, plan: PhasePlan) -> int:
    if action not in plan.green_phases:
        raise ValueError(f'Invalid action {action}; valid={sorted(plan.green_phases)}')
    return plan.green_phases[action]

def transition_phase(current_action: int, next_action: int, plan: PhasePlan) -> tuple[int|None,int]:
    if current_action == next_action: return None, action_to_phase(next_action, plan)
    return plan.yellow_phases.get((current_action,next_action)), action_to_phase(next_action, plan)
