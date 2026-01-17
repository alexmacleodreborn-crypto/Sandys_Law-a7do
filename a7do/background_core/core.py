from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from shared.events import Event, EventType, internal, system_event
from shared.memory import MemoryStore


# ============================================================
# Background Core State
# ============================================================

@dataclass
class BackgroundState:
    """
    Minimal persistent internal state for Background Core.

    These are regulatory variables, not emotions.
    """

    arousal: float = 0.15
    valence: float = 0.0
    confidence: float = 0.10
    confidence_floor: float = 0.05
    uncertainty: float = 0.85
    curiosity: float = 0.25

    # Cursor + counters
    last_seen_seq: int = 0
    cycles: int = 0

    # Stability tracking
    consecutive_high_arousal_cycles: int = 0
    consecutive_low_recovery_cycles: int = 0


# ============================================================
# Background Core Config
# ============================================================

@dataclass(frozen=True)
class BackgroundConfig:
    # Bounds
    arousal_min: float = 0.02
    arousal_max: float = 1.00
    confidence_min: float = 0.02
    confidence_max: float = 1.00
    uncertainty_min: float = 0.00
    uncertainty_max: float = 1.00
    curiosity_min: float = 0.00
    curiosity_max: float = 1.00
    valence_min: float = -1.00
    valence_max: float = 1.00

    # Regulation rates
    arousal_decay: float = 0.03
    uncertainty_decay_on_experience: float = 0.02
    curiosity_rise_on_stagnation: float = 0.03
    curiosity_decay_on_novelty: float = 0.02

    # Surprise impact
    arousal_bump_on_observation: float = 0.015
    arousal_bump_on_failure: float = 0.06
    confidence_drop_on_failure: float = 0.04
    confidence_rise_on_success: float = 0.02

    # Confidence floor
    confidence_floor_rise_on_recovery: float = 0.01
    confidence_floor_max: float = 0.60

    # Cycle gating
    cycle_event_threshold: int = 10
    stagnation_event_threshold: int = 0

    # Burnout guards
    high_arousal_threshold: float = 0.85
    low_recovery_threshold: float = 0.12
    high_arousal_cycle_limit: int = 6
    low_recovery_cycle_limit: int = 8


# ============================================================
# Background Core
# ============================================================

class BackgroundCore:
    """
    Always-on cognitive regulator.

    Uses event counts, not time.
    Emits INTERNAL and SYSTEM events only.
    """

    def __init__(
        self,
        memory: MemoryStore,
        config: Optional[BackgroundConfig] = None,
    ) -> None:
        self.memory = memory
        self.cfg = config or BackgroundConfig()

        # REQUIRED: explicit state creation
        self.state = BackgroundState()

        # Start reading AFTER existing memory
        self.state.last_seen_seq = self.memory.last_seq()

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def step(self) -> List[Event]:
        """
        Advance background regulation if new events exist.
        """
        new_events = list(self.memory.iter_since(self.state.last_seen_seq))
        new_count = len(new_events)

        if new_events:
            self.state.last_seen_seq = new_events[-1][0]

        if new_count >= self.cfg.cycle_event_threshold:
            emitted = self._cycle(new_events, stagnation=False)
        elif new_count <= self.cfg.stagnation_event_threshold:
            emitted = self._cycle([], stagnation=True)
        else:
            emitted = self._light_tick(new_count)

        if emitted:
            self.memory.append_many(emitted)

        return emitted

    # --------------------------------------------------------
    # Regulation Logic
    # --------------------------------------------------------

    def _cycle(
        self,
        new_events: List[Tuple[int, Event]],
        stagnation: bool,
    ) -> List[Event]:
        prev = self._snapshot()
        signal = self._extract_signals(new_events)

        if stagnation:
            self._apply_stagnation()
        else:
            self._apply_experience(signal)

        self._clamp_all()
        self._apply_confidence_floor_logic(prev)
        warnings = self._burnout_guard(prev)

        emitted = [
            internal(
                source="background_core",
                name="state_update",
                payload={
                    "cycles": self.state.cycles,
                    "stagnation": stagnation,
                    "signal": signal,
                    "state": self._snapshot(),
                },
                confidence=self.state.confidence,
            )
        ]

        emitted.extend(warnings)
        self.state.cycles += 1
        return emitted

    def _light_tick(self, new_count: int) -> List[Event]:
        prev = self._snapshot()

        self.state.arousal -= self.cfg.arousal_decay * 0.5

        if new_count > 0:
            self.state.uncertainty -= self.cfg.uncertainty_decay_on_experience * 0.25
            self.state.curiosity -= self.cfg.curiosity_decay_on_novelty * 0.25
        else:
            self.state.curiosity += self.cfg.curiosity_rise_on_stagnation * 0.25

        if self.state.confidence < self.state.confidence_floor:
            self.state.confidence = self.state.confidence_floor
        else:
            self.state.confidence -= 0.005
            if self.state.confidence < self.state.confidence_floor:
                self.state.confidence = self.state.confidence_floor

        self._clamp_all()

        return [
            internal(
                source="background_core",
                name="light_tick",
                payload={
                    "new_event_count": new_count,
                    "delta": self._delta(prev),
                    "state": self._snapshot(),
                },
                confidence=self.state.confidence,
            )
        ]

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _extract_signals(
        self,
        new_events: List[Tuple[int, Event]],
    ) -> Dict[str, float]:
        obs = act = outc = intr = sys = 0
        failures = successes = 0

        for _, e in new_events:
            if e.type == EventType.OBSERVATION:
                obs += 1
            elif e.type == EventType.ACTION:
                act += 1
            elif e.type == EventType.OUTCOME:
                outc += 1
                ok = e.payload.get("ok")
                if ok is True:
                    successes += 1
                elif ok is False:
                    failures += 1
            elif e.type == EventType.INTERNAL:
                intr += 1
            elif e.type == EventType.SYSTEM:
                sys += 1

        total = max(1, len(new_events))
        failure_rate = failures / max(1, failures + successes)

        return {
            "new_total": float(len(new_events)),
            "obs": float(obs),
            "act": float(act),
            "outcome": float(outc),
            "internal": float(intr),
            "system": float(sys),
            "failure_rate": float(failure_rate),
            "activity_ratio": float((obs + act + outc) / total),
        }

    def _apply_experience(self, signal: Dict[str, float]) -> None:
        self.state.arousal += self.cfg.arousal_bump_on_observation * (
            signal["obs"] / max(1.0, signal["new_total"])
        )

        if signal["failure_rate"] > 0:
            self.state.arousal += self.cfg.arousal_bump_on_failure * signal["failure_rate"]
            self.state.confidence -= self.cfg.confidence_drop_on_failure * signal["failure_rate"]
            self.state.uncertainty += 0.03 * signal["failure_rate"]

        success = max(0.0, 1.0 - signal["failure_rate"])
        self.state.confidence += self.cfg.confidence_rise_on_success * success
        self.state.uncertainty -= self.cfg.uncertainty_decay_on_experience * success
        self.state.curiosity -= self.cfg.curiosity_decay_on_novelty * signal["activity_ratio"]
        self.state.arousal -= self.cfg.arousal_decay

    def _apply_stagnation(self) -> None:
        self.state.curiosity += self.cfg.curiosity_rise_on_stagnation
        self.state.arousal -= self.cfg.arousal_decay
        self.state.uncertainty += 0.01
        if self.state.confidence > self.state.confidence_floor:
            self.state.confidence -= 0.01

    def _apply_confidence_floor_logic(self, prev: Dict[str, float]) -> None:
        if self.state.confidence < self.state.confidence_floor:
            self.state.confidence = self.state.confidence_floor

        if self.state.arousal < prev["arousal"] and self.state.confidence > prev["confidence"]:
            self.state.confidence_floor += self.cfg.confidence_floor_rise_on_recovery
            self.state.confidence_floor = min(
                self.state.confidence_floor,
                self.cfg.confidence_floor_max,
            )

    def _burnout_guard(self, prev: Dict[str, float]) -> List[Event]:
        warnings: List[Event] = []

        if self.state.arousal >= self.cfg.high_arousal_threshold:
            self.state.consecutive_high_arousal_cycles += 1
        else:
            self.state.consecutive_high_arousal_cycles = 0

        gain = self.state.confidence - prev["confidence"]
        if gain < self.cfg.low_recovery_threshold and self.state.arousal >= prev["arousal"]:
            self.state.consecutive_low_recovery_cycles += 1
        else:
            self.state.consecutive_low_recovery_cycles = 0

        if self.state.consecutive_high_arousal_cycles >= self.cfg.high_arousal_cycle_limit:
            warnings.append(
                system_event(
                    source="background_core",
                    name="warning_high_arousal_persistence",
                    payload={
                        "cycles": self.state.cycles,
                        "arousal": self.state.arousal,
                    },
                )
            )

        if self.state.consecutive_low_recovery_cycles >= self.cfg.low_recovery_cycle_limit:
            warnings.append(
                system_event(
                    source="background_core",
                    name="warning_low_recovery_persistence",
                    payload={
                        "cycles": self.state.cycles,
                        "confidence": self.state.confidence,
                        "confidence_floor": self.state.confidence_floor,
                    },
                )
            )

        return warnings

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------

    def _clamp(self, v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _clamp_all(self) -> None:
        self.state.arousal = self._clamp(self.state.arousal, self.cfg.arousal_min, self.cfg.arousal_max)
        self.state.confidence = self._clamp(self.state.confidence, self.cfg.confidence_min, self.cfg.confidence_max)
        self.state.confidence_floor = self._clamp(self.state.confidence_floor, 0.0, self.cfg.confidence_floor_max)
        self.state.uncertainty = self._clamp(self.state.uncertainty, self.cfg.uncertainty_min, self.cfg.uncertainty_max)
        self.state.curiosity = self._clamp(self.state.curiosity, self.cfg.curiosity_min, self.cfg.curiosity_max)
        self.state.valence = self._clamp(self.state.valence, self.cfg.valence_min, self.cfg.valence_max)

    def _snapshot(self) -> Dict[str, float]:
        return {
            "arousal": self.state.arousal,
            "valence": self.state.valence,
            "confidence": self.state.confidence,
            "confidence_floor": self.state.confidence_floor,
            "uncertainty": self.state.uncertainty,
            "curiosity": self.state.curiosity,
        }

    def _delta(self, prev: Dict[str, float]) -> Dict[str, float]:
        cur = self._snapshot()
        return {k: cur[k] - prev[k] for k in cur}
