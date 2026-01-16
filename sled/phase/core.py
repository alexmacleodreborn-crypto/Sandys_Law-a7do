from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from shared.events import Event, EventType


# ============================================================
# Phase Snapshot (Advisory Only)
# ============================================================

@dataclass(frozen=True)
class PhaseSnapshot:
    """
    Advisory snapshot of phase / entropy structure.

    This is NOT a decision engine.
    This is NOT control logic.
    It reports patterns only.
    """

    event_count: int

    # Density & structure
    internal_density: float      # ratio of INTERNAL events
    action_density: float        # ratio of ACTION events
    observation_density: float   # ratio of OBSERVATION events

    # Phase signals
    clustering: float            # [0..1]
    volatility: float            # [0..1]
    coherence: float             # [0..1]

    # Transition hints
    phase_state: str             # "stable", "crowding", "transition", "stagnant"
    notes: List[str]


# ============================================================
# SLED Phase Analyzer
# ============================================================

class PhaseAnalyzer:
    """
    SLED — Structure, Load, Entropy Detector.

    Doctrine:
    - Advisory only
    - Deterministic
    - Stateless
    - Event-structure based
    """

    def analyze(self, events: Iterable[Event]) -> PhaseSnapshot:
        events_list = list(events)
        n = len(events_list)

        if n == 0:
            return PhaseSnapshot(
                event_count=0,
                internal_density=0.0,
                action_density=0.0,
                observation_density=0.0,
                clustering=0.0,
                volatility=0.0,
                coherence=1.0,
                phase_state="stagnant",
                notes=["no_events"],
            )

        # ----------------------------
        # Basic densities
        # ----------------------------

        internal = sum(1 for e in events_list if e.type == EventType.INTERNAL)
        action = sum(1 for e in events_list if e.type == EventType.ACTION)
        observation = sum(1 for e in events_list if e.type == EventType.OBSERVATION)

        internal_density = internal / n
        action_density = action / n
        observation_density = observation / n

        # ----------------------------
        # Structural measures
        # ----------------------------

        clustering = self._compute_clustering(events_list)
        volatility = self._compute_volatility(events_list)
        coherence = self._compute_coherence(internal_density, volatility)

        # ----------------------------
        # Phase classification
        # ----------------------------

        phase_state = "stable"
        notes: List[str] = []

        if internal_density > 0.55 and clustering > 0.6:
            phase_state = "crowding"
            notes.append("internal_crowding")

        if volatility > 0.7:
            phase_state = "transition"
            notes.append("high_volatility")

        if action_density < 0.05 and observation_density < 0.05:
            phase_state = "stagnant"
            notes.append("low_external_activity")

        if coherence < 0.3:
            notes.append("low_coherence")

        return PhaseSnapshot(
            event_count=n,
            internal_density=internal_density,
            action_density=action_density,
            observation_density=observation_density,
            clustering=clustering,
            volatility=volatility,
            coherence=coherence,
            phase_state=phase_state,
            notes=notes,
        )

    # ----------------------------
    # Internal computations
    # ----------------------------

    def _compute_clustering(self, events: List[Event]) -> float:
        """
        Clustering: repeated INTERNAL events without intervening resolution.
        """
        if len(events) < 5:
            return 0.0

        internal_runs = 0
        current_run = 0

        for e in events:
            if e.type == EventType.INTERNAL:
                current_run += 1
            else:
                if current_run >= 2:
                    internal_runs += 1
                current_run = 0

        if current_run >= 2:
            internal_runs += 1

        return min(1.0, internal_runs / max(1, len(events) / 5))

    def _compute_volatility(self, events: List[Event]) -> float:
        """
        Volatility: rapid alternation between INTERNAL and ACTION/OUTCOME.
        """
        if len(events) < 6:
            return 0.0

        flips = 0
        last_type: Optional[EventType] = None

        for e in events:
            if last_type is not None and e.type != last_type:
                flips += 1
            last_type = e.type

        return min(1.0, flips / max(1, len(events) - 1))

    def _compute_coherence(self, internal_density: float, volatility: float) -> float:
        """
        Coherence: stability of internal processing relative to volatility.
        """
        # High volatility and high internal density reduce coherence
        coherence = 1.0
        coherence -= 0.6 * volatility
        coherence -= 0.4 * internal_density
        return max(0.0, min(1.0, coherence))
