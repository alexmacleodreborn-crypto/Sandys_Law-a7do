from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from shared.events import Event, EventType, action
from shared.memory import MemoryStore


# ============================================================
# Agent State (Minimal)
# ============================================================

@dataclass
class AgentState:
    """
    Minimal cognition-facing state.

    This is NOT the Background Core state.
    This tracks simple recent experience for action biasing.
    """

    last_action_failed: bool = False
    last_move: Optional[Tuple[int, int]] = None
    steps_since_action: int = 0


# ============================================================
# A7DO Agent (Minimal Cognition Shell)
# ============================================================

class A7DOAgent:
    """
    Minimal decision shell for A7DO.

    Responsibilities:
    - Observe recent events
    - Bias actions based on internal pressure
    - Emit simple ACTION events

    Forbidden:
    - semantics
    - planning
    - goal construction
    """

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory
        self.state = AgentState()

    # ----------------------------
    # Public API
    # ----------------------------

    def decide(self) -> Optional[Event]:
        """
        Decide on the next action (or None).

        This is called after the Background Core step.
        """
        recent = self._recent_events(20)

        internal = self._latest_internal_state(recent)
        if not internal:
            return None

        curiosity = internal.get("curiosity", 0.0)
        arousal = internal.get("arousal", 0.0)
        confidence = internal.get("confidence", 0.0)

        # ----------------------------
        # Action gating (very conservative)
        # ----------------------------

        # If arousal is high, prefer no action (rest)
        if arousal > 0.75:
            self.state.steps_since_action += 1
            return None

        # If curiosity is low and confidence is low, wait
        if curiosity < 0.15 and confidence < 0.2:
            self.state.steps_since_action += 1
            return None

        # ----------------------------
        # Movement selection
        # ----------------------------

        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        # Avoid repeating last failed move
        if self.state.last_action_failed and self.state.last_move in moves:
            moves.remove(self.state.last_move)

        if not moves:
            return None

        dx, dy = random.choice(moves)

        act = action(
            source="a7do",
            name="move",
            payload={"dx": dx, "dy": dy},
        )

        self.state.last_move = (dx, dy)
        self.state.steps_since_action = 0

        return act

    # ----------------------------
    # Observation Helpers
    # ----------------------------

    def observe_outcomes(self) -> None:
        """
        Update internal bias state from recent outcomes.
        """
        recent = self._recent_events(10)
        for _, e in reversed(recent):
            if e.type == EventType.OUTCOME and e.parent_id:
                ok = e.payload.get("ok", None)
                if ok is False:
                    self.state.last_action_failed = True
                elif ok is True:
                    self.state.last_action_failed = False
                break

    # ----------------------------
    # Utilities
    # ----------------------------

    def _recent_events(self, n: int) -> List[Tuple[int, Event]]:
        start = max(0, self.memory.last_seq() - n)
        return list(self.memory.iter_since(start))

    def _latest_internal_state(self, recent: List[Tuple[int, Event]]) -> Optional[Dict[str, float]]:
        for _, e in reversed(recent):
            if e.type == EventType.INTERNAL and e.name == "state_update":
                return e.payload.get("state", {})
        return None
