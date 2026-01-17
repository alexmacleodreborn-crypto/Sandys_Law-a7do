from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from shared.events import Event, EventType, internal


# ============================================================
# Boundary State
# ============================================================

@dataclass
class BoundaryState:
    """
    Tracks locations where movement repeatedly fails.

    We do NOT store walls.
    We store experienced resistance.
    """
    hits: Dict[Tuple[int, int], int]


# ============================================================
# Boundary Detector
# ============================================================

class BoundaryDetector:
    """
    Detects continuous spatial boundaries from prediction errors.

    Doctrine:
    - No geometry
    - No labels
    - No objects
    - Only repeated resistance
    """

    def __init__(self, threshold: int = 2) -> None:
        self.state = BoundaryState(hits={})
        self.threshold = int(threshold)

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def observe(self, events: List[Event]) -> List[Event]:
        """
        Observe prediction errors and detect boundary formation.
        """
        emitted: List[Event] = []

        for e in events:
            if e.type != EventType.INTERNAL:
                continue

            if e.name != "prediction_error":
                continue

            # We only care about movement-related prediction errors
            expected = e.payload.get("expected", {})
            observed = e.payload.get("observed", {})

            # World typically reports attempted + resulting position
            pos = observed.get("position")
            if not pos or not isinstance(pos, (list, tuple)):
                continue

            x, y = int(pos[0]), int(pos[1])
            key = (x, y)

            # Accumulate resistance
            self.state.hits[key] = self.state.hits.get(key, 0) + 1

            # If threshold crossed, emit boundary signal
            if self.state.hits[key] == self.threshold:
                emitted.append(
                    internal(
                        source="boundary",
                        name="boundary_detected",
                        payload={
                            "position": {"x": x, "y": y},
                            "hits": self.state.hits[key],
                        },
                        confidence=1.0,
                    )
                )

        return emitted