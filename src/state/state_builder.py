from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class StateSpec:
    local_incoming: tuple[str, ...]
    neighbour_incoming: tuple[str, ...]
    neighbour_actions: tuple[str, ...]
    max_queue: float = 40.0

class StateBuilder:
    """Builds Chapter-4 semi-coordinated state S(t)=[Q(t), Q'(t), a'(t)]."""
    def __init__(self, spec: StateSpec): self.spec = spec
    @property
    def dimension(self) -> int:
        return len(self.spec.local_incoming)+len(self.spec.neighbour_incoming)+len(self.spec.neighbour_actions)
    def build(self, queues: dict[str, float], neighbour_actions: dict[str, int]) -> list[float]:
        vals=[]
        for edge in (*self.spec.local_incoming, *self.spec.neighbour_incoming):
            vals.append(min(max(float(queues.get(edge,0.0)),0.0), self.spec.max_queue)/self.spec.max_queue)
        for sid in self.spec.neighbour_actions:
            vals.append(float(neighbour_actions.get(sid,0)))
        return vals
