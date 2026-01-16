from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from shared.events import Event, observation, outcome


# ============================================================
# World Primitives (No Semantics)
# ============================================================

@dataclass(frozen=True)
class WorldObject:
    """
    Objective object in the world.
    No meaning, no label, no intent.
    """
    id: str
    solid: bool = True


@dataclass
class AgentBody:
    """
    Physical presence of an agent in the world.
    The world does not know about cognition.
    """
    x: int
    y: int


# ============================================================
# World Grid
# ============================================================

class World:
    """
    Objective grid-based world.

    Responsibilities:
    - maintain spatial state
    - apply movement physics
    - detect collisions
    - emit observation and outcome events

    Forbidden:
    - semantics
    - labels
    - goals
    - emotions
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)

        self.objects: Dict[Tuple[int, int], WorldObject] = {}
        self.agent: Optional[AgentBody] = None

    # ----------------------------
    # Setup
    # ----------------------------

    def spawn_agent(self, x: int, y: int) -> List[Event]:
        """
        Place the agent body in the world.
        """
        self.agent = AgentBody(x=int(x), y=int(y))
        return [
            observation(
                source="world",
                name="agent_spawned",
                payload={"x": self.agent.x, "y": self.agent.y},
            )
        ]

    def add_object(self, x: int, y: int, solid: bool = True) -> None:
        """
        Add an object at a grid location.
        """
        self.objects[(int(x), int(y))] = WorldObject(
            id=str(uuid4()),
            solid=bool(solid),
        )

    # ----------------------------
    # World Step
    # ----------------------------

    def step(self, action_event: Event) -> List[Event]:
        """
        Apply an ACTION event and return resulting world events.

        The world only understands:
        - action.name
        - action.payload
        """
        if not self.agent:
            return []

        if action_event.name == "move":
            return self._apply_move(action_event)

        # Unknown actions are ignored but recorded as failed outcomes
        return [
            outcome(
                source="world",
                name="unknown_action",
                payload={"ok": False, "reason": "unrecognized_action"},
                parent_id=action_event.id,
            )
        ]

    # ----------------------------
    # Physics
    # ----------------------------

    def _apply_move(self, action_event: Event) -> List[Event]:
        """
        Attempt to move the agent by dx, dy.
        """
        dx = int(action_event.payload.get("dx", 0))
        dy = int(action_event.payload.get("dy", 0))

        target_x = self.agent.x + dx
        target_y = self.agent.y + dy

        events: List[Event] = []

        # Boundary check
        if not self._in_bounds(target_x, target_y):
            events.append(
                observation(
                    source="world",
                    name="boundary_contact",
                    payload={
                        "from": (self.agent.x, self.agent.y),
                        "attempt": (target_x, target_y),
                    },
                )
            )
            events.append(
                outcome(
                    source="world",
                    name="move_blocked",
                    payload={"ok": False, "reason": "out_of_bounds"},
                    parent_id=action_event.id,
                )
            )
            return events

        # Object collision
        obj = self.objects.get((target_x, target_y))
        if obj and obj.solid:
            events.append(
                observation(
                    source="world",
                    name="collision",
                    payload={
                        "object_id": obj.id,
                        "at": (target_x, target_y),
                    },
                )
            )
            events.append(
                outcome(
                    source="world",
                    name="move_blocked",
                    payload={"ok": False, "reason": "collision"},
                    parent_id=action_event.id,
                )
            )
            return events

        # Movement succeeds
        self.agent.x = target_x
        self.agent.y = target_y

        events.append(
            observation(
                source="world",
                name="position",
                payload={"x": self.agent.x, "y": self.agent.y},
            )
        )
        events.append(
            outcome(
                source="world",
                name="move_ok",
                payload={"ok": True},
                parent_id=action_event.id,
            )
        )

        return events

    # ----------------------------
    # Utilities
    # ----------------------------

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def snapshot(self) -> Dict[str, object]:
        """
        Debug-only snapshot of world state.
        Never used for cognition.
        """
        return {
            "agent": None if not self.agent else (self.agent.x, self.agent.y),
            "objects": {k: v.id for k, v in self.objects.items()},
            "size": (self.width, self.height),
        }
