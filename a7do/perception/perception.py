from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from shared.events import Event, EventType


# ============================================================
# Percept Types (No Semantics)
# ============================================================

class PerceptType(str, Enum):
    POSITION = "position"
    CONTACT = "contact"
    BOUNDARY = "boundary"
    ACTION_RESULT = "action_result"
    UNKNOWN = "unknown"


# ============================================================
# Percept Object
# ============================================================

@dataclass(frozen=True)
class Percept:
    """
    A structured interpretation of an event, still non-semantic.

    Example:
    - A collision event becomes a CONTACT percept with object_id + location.
    - A move outcome becomes an ACTION_RESULT percept with ok/reason.
    """
    type: PerceptType
    source_event_id: str
    payload: Dict[str, Any]


# ============================================================
# Perception Engine
# ============================================================

class PerceptionEngine:
    """
    Converts raw events into percepts.

    Design rules:
    - No labels like "door", "tv"
    - No inference beyond structural typing
    - Deterministic mapping where possible
    """

    def __init__(self) -> None:
        self._last_position: Optional[Tuple[int, int]] = None

    def process(self, events: List[Event]) -> List[Percept]:
        percepts: List[Percept] = []

        for e in events:
            if e.type == EventType.OBSERVATION:
                percepts.extend(self._from_observation(e))
            elif e.type == EventType.OUTCOME:
                percepts.append(self._from_outcome(e))
            else:
                # other event types are not perceptual input
                continue

        return percepts

    # ----------------------------
    # Internal mappers
    # ----------------------------

    def _from_observation(self, e: Event) -> List[Percept]:
        out: List[Percept] = []

        if e.name == "position":
            x = int(e.payload.get("x"))
            y = int(e.payload.get("y"))
            self._last_position = (x, y)
            out.append(
                Percept(
                    type=PerceptType.POSITION,
                    source_event_id=e.id,
                    payload={"x": x, "y": y},
                )
            )
            return out

        if e.name == "collision":
            # contact with a solid object
            out.append(
                Percept(
                    type=PerceptType.CONTACT,
                    source_event_id=e.id,
                    payload={
                        "object_id": e.payload.get("object_id"),
                        "at": e.payload.get("at"),
                    },
                )
            )
            return out

        if e.name == "boundary_contact":
            out.append(
                Percept(
                    type=PerceptType.BOUNDARY,
                    source_event_id=e.id,
                    payload={
                        "from": e.payload.get("from"),
                        "attempt": e.payload.get("attempt"),
                    },
                )
            )
            return out

        # Unknown observation type
        out.append(
            Percept(
                type=PerceptType.UNKNOWN,
                source_event_id=e.id,
                payload={"name": e.name, "payload": dict(e.payload)},
            )
        )
        return out

    def _from_outcome(self, e: Event) -> Percept:
        ok = e.payload.get("ok", None)
        reason = e.payload.get("reason", None)

        return Percept(
            type=PerceptType.ACTION_RESULT,
            source_event_id=e.id,
            payload={
                "ok": ok,
                "reason": reason,
                "parent_id": e.parent_id,
                "name": e.name,
            },
        )

    # ----------------------------
    # Introspection helpers (optional)
    # ----------------------------

    def last_position(self) -> Optional[Tuple[int, int]]:
        return self._last_position
