from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4


# ============================================================
# Event Types
# ============================================================

class EventType(str, Enum):
    """
    High-level classification of events.

    These are intentionally abstract so the same event system
    can support World, A7DO cognition, Background Core, and SLED.
    """
    OBSERVATION = "observation"   # Something perceived
    ACTION = "action"             # Something attempted
    OUTCOME = "outcome"           # Result of an action
    INTERNAL = "internal"         # Thought, emotion shift, drive change
    SYSTEM = "system"             # Startup, shutdown, warnings


# ============================================================
# Base Event
# ============================================================

@dataclass(frozen=True)
class Event:
    """
    Canonical event object shared across the entire system.

    Events are immutable. They represent *what happened*,
    not interpretations of meaning.
    """

    type: EventType
    source: str                   # e.g. "world", "a7do", "background_core"
    name: str                     # short semantic label (e.g. "touch", "move")
    payload: Dict[str, Any]       # event-specific data

    id: str = field(default_factory=lambda: str(uuid4()))
    parent_id: Optional[str] = None   # causal linkage
    confidence: Optional[float] = None  # optional belief strength (0–1)

    def summary(self) -> str:
        """
        Human-readable one-line summary for logging / debugging.
        """
        return f"[{self.type}] {self.source}:{self.name} ({self.id})"


# ============================================================
# Convenience Constructors
# ============================================================

def observation(
    *,
    source: str,
    name: str,
    payload: Dict[str, Any],
    confidence: Optional[float] = None,
    parent_id: Optional[str] = None,
) -> Event:
    return Event(
        type=EventType.OBSERVATION,
        source=source,
        name=name,
        payload=payload,
        confidence=confidence,
        parent_id=parent_id,
    )


def action(
    *,
    source: str,
    name: str,
    payload: Dict[str, Any],
    parent_id: Optional[str] = None,
) -> Event:
    return Event(
        type=EventType.ACTION,
        source=source,
        name=name,
        payload=payload,
        parent_id=parent_id,
    )


def outcome(
    *,
    source: str,
    name: str,
    payload: Dict[str, Any],
    parent_id: Optional[str] = None,
) -> Event:
    return Event(
        type=EventType.OUTCOME,
        source=source,
        name=name,
        payload=payload,
        parent_id=parent_id,
    )


def internal(
    *,
    source: str,
    name: str,
    payload: Dict[str, Any],
    confidence: Optional[float] = None,
    parent_id: Optional[str] = None,
) -> Event:
    return Event(
        type=EventType.INTERNAL,
        source=source,
        name=name,
        payload=payload,
        confidence=confidence,
        parent_id=parent_id,
    )


def system_event(
    *,
    source: str,
    name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Event:
    return Event(
        type=EventType.SYSTEM,
        source=source,
        name=name,
        payload=payload or {},
    )

