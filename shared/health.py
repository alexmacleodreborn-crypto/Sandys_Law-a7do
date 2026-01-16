from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from shared.events import Event, EventType


# ============================================================
# Health Signals (Computed, Not Stored)
# ============================================================

@dataclass(frozen=True)
class HealthSnapshot:
    """
    Advisory snapshot of system stability.

    This is NOT control logic.
    This is NOT a decision engine.
    It reports signals that other systems may consume.
    """

    event_count: int

    # Emotional / regulatory indicators
    arousal: Optional[float]
    confidence: Optional[float]
    confidence_floor: Optional[float]
    uncertainty: Optional[float]
    curiosity: Optional[float]

    # Derived risks
    zeno_risk: float            # [0..1]
    burnout_risk: float         # [0..1]
    stagnation_risk: float      # [0..1]

    notes: List[str]


# ============================================================
# Health Analyzer
# ============================================================

class HealthAnalyzer:
    """
    Computes health metrics from an event stream.

    Doctrine:
    - Purely advisory
    - Deterministic
    - No mutation
    - No semantics
    """

    def __init__(self) -> None:
        pass

    # ----------------------------
    # Public API
    # ----------------------------

    def analyze(self, events: Iterable[Event]) -> HealthSnapshot:
        events_list = list(events)
        n = len(events_list)

        # Extract last known internal state
        internal_state = self._latest_internal_state(events_list)

        arousal = internal_state.get("arousal")
        confidence = internal_state.get("confidence")
        confidence_floor = internal_state.get("confidence_floor")
        uncertainty = internal_state.get("uncertainty")
        curiosity = internal_state.get("curiosity")

        # Compute risks
        zeno = self._compute_zeno_risk(events_list, arousal, confidence)
        burnout = self._compute_burnout_risk(arousal, confidence, confidence_floor)
        stagnation = self._compute_stagnation_risk(events_list, curiosity)

        notes: List[str] = []
        if zeno > 0.7:
            notes.append("elevated_zeno_risk")
        if burnout > 0.7:
            notes.append("elevated_burnout_risk")
        if stagnation > 0.7:
            notes.append("elevated_stagnation_risk")

        return HealthSnapshot(
            event_count=n,
            arousal=arousal,
            confidence=confidence,
            confidence_floor=confidence_floor,
            uncertainty=uncertainty,
            curiosity=curiosity,
            zeno_risk=zeno,
            burnout_risk=burnout,
            stagnation_risk=stagnation,
            notes=notes,
        )

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _latest_internal_state(self, events: List[Event]) -> Dict[str, Optional[float]]:
        """
        Finds the most recent Background Core state_update.
        """
        for e in reversed(events):
            if e.type == EventType.INTERNAL and e.name == "state_update":
                state = e.payload.get("state", {})
                return {
                    "arousal": state.get("arousal"),
                    "confidence": state.get("confidence"),
                    "confidence_floor": state.get("confidence_floor"),
                    "uncertainty": state.get("uncertainty"),
                    "curiosity": state.get("curiosity"),
                }

        return {
            "arousal": None,
            "confidence": None,
            "confidence_floor": None,
            "uncertainty": None,
            "curiosity": None,
        }

    def _compute_zeno_risk(
        self,
        events: List[Event],
        arousal: Optional[float],
        confidence: Optional[float],
    ) -> float:
        """
        Zeno risk: rapid cycling without resolution.
        """
        if arousal is None or confidence is None:
            return 0.0

        internal_events = [e for e in events if e.type == EventType.INTERNAL]

        if len(internal_events) < 5:
            return 0.0

        # High arousal + low confidence + dense internal cycling
        density = min(1.0, len(internal_events) / max(1, len(events)))
        risk = 0.0

        if arousal > 0.8 and confidence < 0.25:
            risk += 0.5

        risk += 0.5 * density
        return min(1.0, risk)

    def _compute_burnout_risk(
        self,
        arousal: Optional[float],
        confidence: Optional[float],
        confidence_floor: Optional[float],
    ) -> float:
        """
        Burnout risk: sustained activation without recovery.
        """
        if arousal is None or confidence is None or confidence_floor is None:
            return 0.0

        risk = 0.0

        if arousal > 0.85:
            risk += 0.4

        if confidence < confidence_floor + 0.05:
            risk += 0.4

        if arousal > 0.9 and confidence < 0.2:
            risk += 0.2

        return min(1.0, risk)

    def _compute_stagnation_risk(
        self,
        events: List[Event],
        curiosity: Optional[float],
    ) -> float:
        """
        Stagnation risk: nothing happening while curiosity rises.
        """
        if curiosity is None:
            return 0.0

        obs_count = sum(1 for e in events if e.type == EventType.OBSERVATION)
        act_count = sum(1 for e in events if e.type == EventType.ACTION)

        if obs_count + act_count == 0 and curiosity > 0.6:
            return min(1.0, curiosity)

        return 0.0

