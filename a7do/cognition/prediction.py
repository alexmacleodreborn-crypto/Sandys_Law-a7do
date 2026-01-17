from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from shared.events import Event, EventType, internal


# ============================================================
# Prediction State
# ============================================================

@dataclass
class PredictionState:
    """
    Minimal predictive memory.

    Stores the last expectation derived from an action.
    """
    expected: Optional[Dict[str, float]] = None


# ============================================================
# Prediction Engine
# ============================================================

class PredictionEngine:
    """
    Computes prediction error and expectation confirmation.

    Doctrine:
    - No rewards
    - No goals
    - No learning rate
    - Pure mismatch / confirmation detection
    """

    def __init__(self) -> None:
        self.state = PredictionState()

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def observe(self, events: List[Event]) -> List[Event]:
        """
        Inspect recent events and emit:
        - prediction_error
        - expectation_confirmed (when appropriate)
        """
        emitted: List[Event] = []

        for e in events:
            # --------------------------------------------
            # Action → expectation
            # --------------------------------------------
            if e.type == EventType.ACTION:
                self.state.expected = self._expect_from_action(e)

            # --------------------------------------------
            # Outcome → resolve expectation
            # --------------------------------------------
            elif e.type == EventType.OUTCOME and self.state.expected is not None:
                error = self._compute_error(self.state.expected, e)

                # Always emit prediction error
                emitted.append(
                    internal(
                        source="prediction",
                        name="prediction_error",
                        payload={
                            "error": error,
                            "expected": self.state.expected,
                            "observed": e.payload,
                        },
                        confidence=max(0.0, 1.0 - min(1.0, error)),
                    )
                )

                # Emit confirmation if error is sufficiently low
                if error < 0.1:
                    emitted.append(
                        internal(
                            source="prediction",
                            name="expectation_confirmed",
                            payload={
                                "expected": self.state.expected,
                                "observed": e.payload,
                            },
                            confidence=1.0,
                        )
                    )

                # Expectation resolved (single-shot)
                self.state.expected = None

        return emitted

    # --------------------------------------------------------
    # Internals
    # --------------------------------------------------------

    def _expect_from_action(self, e: Event) -> Dict[str, float]:
        """
        Convert an action into a minimal numeric expectation.

        Currently supports movement only.
        """
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
        L1 error between expected and observed outcome.
        """
        err = 0.0
        payload = outcome.payload or {}

        for k, v in expected.items():
            observed = float(payload.get(k, 0.0))
            err += abs(v - observed)

        return float(err)