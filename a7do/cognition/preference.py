from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from shared.events import Event, EventType, internal


# ============================================================
# Preference State
# ============================================================

@dataclass
class PreferenceState:
    """
    Stores emergent preferences.

    Keyed by simple feature signatures (e.g. positions, actions).
    """
    scores: Dict[str, float]


# ============================================================
# Preference Engine
# ============================================================

class PreferenceEngine:
    """
    Forms like / not-like biases from confirmed expectations.

    Doctrine:
    - No rewards
    - No goals
    - No emotions
    - Pure bias accumulation
    """

    def __init__(self, learning_rate: float = 0.05) -> None:
        self.state = PreferenceState(scores={})
        self.lr = float(learning_rate)

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def observe(self, events: List[Event]) -> List[Event]:
        emitted: List[Event] = []

        for e in events:
            if e.type != EventType.INTERNAL:
                continue

            # --------------------------------------------
            # Positive preference formation
            # --------------------------------------------
            if e.name == "expectation_confirmed":
                key = self._key_from_event(e)
                if not key:
                    continue

                prev = self.state.scores.get(key, 0.0)
                new = prev + self.lr
                self.state.scores[key] = new

                emitted.append(
                    internal(
                        source="preference",
                        name="preference_updated",
                        payload={
                            "key": key,
                            "delta": +self.lr,
                            "score": new,
                        },
                        confidence=min(1.0, new),
                    )
                )

            # --------------------------------------------
            # Negative preference (persistent surprise)
            # --------------------------------------------
            elif e.name == "prediction_error":
                error = float(e.payload.get("error", 0.0))
                if error < 0.3:
                    continue

                key = self._key_from_event(e)
                if not key:
                    continue

                prev = self.state.scores.get(key, 0.0)
                new = prev - self.lr * error
                self.state.scores[key] = new

                emitted.append(
                    internal(
                        source="preference",
                        name="preference_updated",
                        payload={
                            "key": key,
                            "delta": -self.lr * error,
                            "score": new,
                        },
                        confidence=max(0.0, 1.0 - error),
                    )
                )

        return emitted

    # --------------------------------------------------------
    # Internals
    # --------------------------------------------------------

    def _key_from_event(self, e: Event) -> Optional[str]:
        """
        Reduce an event to a stable preference key.

        For now:
        - movement preference by resulting position
        """
        obs = e.payload.get("observed", {})
        pos = obs.get("position")
        if pos and isinstance(pos, (list, tuple)):
            return f"pos:{int(pos[0])},{int(pos[1])}"

        return None