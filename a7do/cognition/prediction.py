from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from shared.events import Event, internal, EventType


# ============================================================
# Prediction State
# ============================================================

@dataclass
class PredictionState:
    """
    Minimal predictive memory.

    Stores last expectation derived from action.
    """
    expected: Optional[Dict[str, float]] = None


# ============================================================
# Prediction Engine
# ============================================================

class PredictionEngine:
    """
    Computes prediction error from action → outcome mismatch.

    Doctrine:
    - No goals
    - No reward
    - No learning rate
    - Pure mismatch detection
    """

    def __init__(self) -> None:
        self.state = PredictionState()

    # ----------------------------
    # Public API
    # ----------------------------

    def observe(self, events: List[Event]) -> List[Event]:
        """
        Inspect recent events and emit prediction_error if applicable.
        """
        emitted: List[Event] = []

        for e in events:
            # 1) Action creates an expectation
            if e.type == EventType.ACTION:
                self.state.expected = self._expect_from_action(e)

            # 2) Outcome resolves expectation
            elif e.type == EventType.OUTCOME and self.state.expected is not None:
                error = self._compute_error(self.state.expected, e)

                emitted.append(
                    internal(
                        source="prediction",
                        name="prediction_error",
                        payload={
                            "error": error,
                            "expected": self.state.expected,
                            "observed": e.payload,
                        },
                        confidence=1.0 - min(1.0, error),
                    )
                )

                # expectation resolved
                self.state.expected = None

        return emitted

    # ----------------------------
    # Internals
    # ----------------------------

    def _expect_from_action(self, e: Event) -> Dict[str, float]:
        """
        Convert an action into a minimal numeric expectation.
        """
        # For now: movement expects change
        if e.name == "move":
            dx = float(e.payload.get("dx", 0))
            dy = float(e.payload.get("dy", 0))
            return {
                "dx": dx,
                "dy": dy,
            }

        return {}

    def _compute_error(self, expected: Dict[str, float], outcome: Event) -> float:
        """
        L1 error between expected and observed.
        """
        err = 0.0
        for k, v in expected.items():
            observed = float(outcome.payload.get(k, 0.0))
            err += abs(v - observed)
        return float(err)
