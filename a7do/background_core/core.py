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

    NOTE:
    - This is not "human emotion".
    - These are regulatory control variables.
    - Values are bounded and change gradually.
    """

    arousal: float = 0.15          # [0..1]
    valence: float = 0.0           # [-1..1] (kept neutral early)
    confidence: float = 0.10       # [0..1]
    confidence_floor: float = 0.05 # [0..1] rises with healthy recovery
    uncertainty: float = 0.85      # [0..1]
    curiosity: float = 0.25        # [0..1]

    # Counters (event-count based)
    last_seen_seq: int = 0
    cycles: int = 0

    # For burnout / zeno detection (event-count based, not time-based)
    consecutive_high_arousal_cycles: int = 0
    consecutive_low_recovery_cycles: int = 0


# ============================================================
# Background Core Config
# ============================================================

@dataclass(frozen=True)
class BackgroundConfig:
    """
    Tunables. Keep these conservative.
    We'll adjust only after behavior is observed.

    All changes should preserve:
    - gradual decay,
    - non-zero confidence recovery,
    - rising confidence floor in healthy cycles.
    """

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

    # Regulation rates (per "cycle", not per second)
    arousal_decay: float = 0.03
    uncertainty_decay_on_experience: float = 0.02
    curiosity_rise_on_stagnation: float = 0.03
    curiosity_decay_on_novelty: float = 0.02

    # Surprise / novelty impact
    arousal_bump_on_observation: float = 0.015
    arousal_bump_on_failure: float = 0.06
    confidence_drop_on_failure: float = 0.04
    confidence_rise_on_success: float = 0.02

    # Confidence floor dynamics (the key doctrine behavior)
    confidence_floor_rise_on_recovery: float = 0.01
    confidence_floor_max: float = 0.60

    # Cycle gating
    cycle_event_threshold: int = 10  # minimum new events before a meaningful cycle
    stagnation_event_threshold: int = 0  # if no new events, treat as stagnation

    # Burnout / Zeno guardrails
    high_arousal_threshold: float = 0.85
    low_recovery_threshold: float = 0.12  # confidence increase too small
    high_arousal_cycle_limit: int = 6
    low_recovery_cycle_limit: int = 8


# ============================================================
# Background Core
# ============================================================

class BackgroundCore:
    """
    Always-on cognitive regulator.

    It does NOT plan or choose actions.
    It only:
    - regulates internal state,
    - emits INTERNAL events,
    - emits SYSTEM warnings when instability is detected.

    It advances using event counts (seq deltas), not time.
    """

    def __init__(self, memory: MemoryStore, config: Optional[BackgroundConfig] = None) -> None:
        self.memory = memory
        self.cfg = config or BackgroundConfig()
        self.state.last_seen_seq = self.memory.last_seq())
        
    # ----------------------------
    # Public API
    # ----------------------------

    def step(self) -> List[Event]:
        """
        Advance the background core by one cycle if needed.

        Returns events emitted this step (also written to memory).
        """
        new_events = list(self.memory.iter_since(self.state.last_seen_seq))
        new_count = len(new_events)
        last_seq = self.state.last_seen_seq

        if new_events:
            last_seq = new_events[-1][0]

        # Decide if we run a meaningful cycle
        if new_count >= self.cfg.cycle_event_threshold:
            emitted = self._cycle(new_events, stagnation=False)
        elif new_count <= self.cfg.stagnation_event_threshold:
            emitted = self._cycle([], stagnation=True)
        else:
            # Not enough new information; do a light regulation tick anyway (still "always-on")
            emitted = self._light_tick(new_count)

        # Update seen seq after processing
        self.state.last_seen_seq = last_seq

        # Persist emitted events (append-only)
        if emitted:
            self.memory.append_many(emitted)

        return emitted

    # ----------------------------
    # Internal logic
    # ----------------------------

    def _cycle(self, new_events: List[Tuple[int, Event]], stagnation: bool) -> List[Event]:
        """
        Full regulation cycle. This is where the doctrine behaviors occur:
        - arousal rises with novelty/surprise, then decays
        - confidence can drop on failure, but recovers to a rising floor
        - curiosity rises under stagnation
        - system warns on runaway patterns
        """
        prev = self._snapshot()

        # 1) Extract simple signals (no semantics)
        signal = self._extract_signals(new_events)

        # 2) Apply regulation
        if stagnation:
            self._apply_stagnation()
        else:
            self._apply_experience(signal)

        # 3) Enforce bounds
        self._clamp_all()

        # 4) Recovery floor logic (key: floor rises after successful stabilization)
        self._apply_confidence_floor_logic(prev)

        # 5) Burnout / Zeno guard
        warnings = self._burnout_guard(prev)

        # 6) Emit internal state update event
        emitted: List[Event] = []
        emitted.append(
            internal(
                source="background_core",
                name="state_update",
                payload={
                    "cycles": self.state.cycles,
                    "stagnation": bool(stagnation),
                    "delta": self._delta(prev),
                    "state": self._snapshot(),
                    "signal": signal,
                },
                confidence=self.state.confidence,
            )
        )

        emitted.extend(warnings)

        self.state.cycles += 1
        return emitted

    def _light_tick(self, new_count: int) -> List[Event]:
        """
        Very small regulation step even when insufficient new events arrive.
        Keeps the system "alive" without forcing big changes.
        """
        prev = self._snapshot()

        # Slight arousal decay
        self.state.arousal -= self.cfg.arousal_decay * 0.5

        # Uncertainty decays slightly only if there is some activity
        if new_count > 0:
            self.state.uncertainty -= self.cfg.uncertainty_decay_on_experience * 0.25
            self.state.curiosity -= self.cfg.curiosity_decay_on_novelty * 0.25
        else:
            self.state.curiosity += self.cfg.curiosity_rise_on_stagnation * 0.25

        # Confidence gently relaxes toward the floor (never below it)
        if self.state.confidence < self.state.confidence_floor:
            self.state.confidence = self.state.confidence_floor
        else:
            # slight drift down, but bounded by floor
            self.state.confidence -= 0.005
            if self.state.confidence < self.state.confidence_floor:
                self.state.confidence = self.state.confidence_floor

        self._clamp_all()

        return [
            internal(
                source="background_core",
                name="light_tick",
                payload={
                    "new_event_count": int(new_count),
                    "delta": self._delta(prev),
                    "state": self._snapshot(),
                },
                confidence=self.state.confidence,
            )
        ]

    def _extract_signals(self, new_events: List[Tuple[int, Event]]) -> Dict[str, float]:
        """
        Convert a set of new events into small numeric signals.

        IMPORTANT:
        - No semantics, no labels.
        - Only generic facts: counts and simple rates.
        """
        obs = 0
        act = 0
        outc = 0
        intr = 0
        sys = 0

        failures = 0
        successes = 0

        for _, e in new_events:
            if e.type == EventType.OBSERVATION:
                obs += 1
            elif e.type == EventType.ACTION:
                act += 1
            elif e.type == EventType.OUTCOME:
                outc += 1
                # convention: outcomes can carry {"ok": bool} in payload
                ok = e.payload.get("ok", None)
                if ok is True:
                    successes += 1
                elif ok is False:
                    failures += 1
            elif e.type == EventType.INTERNAL:
                intr += 1
            elif e.type == EventType.SYSTEM:
                sys += 1

        total = max(1, len(new_events))
        failure_rate = failures / max(1, (failures + successes))
        activity = (obs + act + outc) / total

        return {
            "new_total": float(len(new_events)),
            "obs": float(obs),
            "act": float(act),
            "outcome": float(outc),
            "internal": float(intr),
            "system": float(sys),
            "failure_rate": float(failure_rate),
            "activity_ratio": float(activity),
        }

    def _apply_experience(self, signal: Dict[str, float]) -> None:
        # novelty/experience bumps arousal modestly
        self.state.arousal += self.cfg.arousal_bump_on_observation * (signal["obs"] / max(1.0, signal["new_total"]))

        # failures increase arousal and reduce confidence
        if signal["failure_rate"] > 0.0:
            self.state.arousal += self.cfg.arousal_bump_on_failure * signal["failure_rate"]
            self.state.confidence -= self.cfg.confidence_drop_on_failure * signal["failure_rate"]
            self.state.uncertainty += 0.03 * signal["failure_rate"]

        # successes slightly raise confidence and reduce uncertainty
        success_inferred = max(0.0, 1.0 - signal["failure_rate"])
        self.state.confidence += self.cfg.confidence_rise_on_success * success_inferred
        self.state.uncertainty -= self.cfg.uncertainty_decay_on_experience * success_inferred

        # curiosity decays when novelty is present
        self.state.curiosity -= self.cfg.curiosity_decay_on_novelty * min(1.0, signal["activity_ratio"])

        # arousal naturally decays after processing
        self.state.arousal -= self.cfg.arousal_decay

    def _apply_stagnation(self) -> None:
        # When nothing happens, curiosity pressure rises
        self.state.curiosity += self.cfg.curiosity_rise_on_stagnation

        # Arousal decays toward rest
        self.state.arousal -= self.cfg.arousal_decay

        # Uncertainty doesn't automatically resolve during stagnation; slight creep upward
        self.state.uncertainty += 0.01

        # Confidence relaxes toward floor (no reason to be "sure" without evidence)
        if self.state.confidence > self.state.confidence_floor:
            self.state.confidence -= 0.01

    def _apply_confidence_floor_logic(self, prev: Dict[str, float]) -> None:
        """
        Doctrine-critical behavior:
        - confidence does not reset to 0 after recovery
        - each healthy stabilization can raise the minimum confidence floor
        """
        # Ensure confidence never below floor
        if self.state.confidence < self.state.confidence_floor:
            self.state.confidence = self.state.confidence_floor

        # If arousal decreased and confidence improved, raise the floor slightly
        arousal_decreased = self.state.arousal < prev["arousal"]
        confidence_increased = self.state.confidence > prev["confidence"]

        if arousal_decreased and confidence_increased:
            self.state.confidence_floor += self.cfg.confidence_floor_rise_on_recovery
            if self.state.confidence_floor > self.cfg.confidence_floor_max:
                self.state.confidence_floor = self.cfg.confidence_floor_max

        # Floor should never exceed current confidence hard bound
        if self.state.confidence_floor > self.state.confidence:
            # allow floor to "catch up" only if confidence is rising next cycles
            self.state.confidence_floor = min(self.state.confidence_floor, self.state.confidence)

    def _burnout_guard(self, prev: Dict[str, float]) -> List[Event]:
        """
        Detect runaway cycles without using time.
        Emits SYSTEM warnings if thresholds are crossed.
        """
        warnings: List[Event] = []

        # high arousal tracking
        if self.state.arousal >= self.cfg.high_arousal_threshold:
            self.state.consecutive_high_arousal_cycles += 1
        else:
            self.state.consecutive_high_arousal_cycles = 0

        # recovery quality tracking (did confidence meaningfully recover?)
        confidence_gain = self.state.confidence - prev["confidence"]
        if confidence_gain < self.cfg.low_recovery_threshold and self.state.arousal >= prev["arousal"]:
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
                        "consecutive": self.state.consecutive_high_arousal_cycles,
                        "arousal": self.state.arousal,
                        "confidence": self.state.confidence,
                        "uncertainty": self.state.uncertainty,
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
                        "consecutive": self.state.consecutive_low_recovery_cycles,
                        "arousal": self.state.arousal,
                        "confidence": self.state.confidence,
                        "confidence_floor": self.state.confidence_floor,
                    },
                )
            )

        return warnings

    # ----------------------------
    # Utilities
    # ----------------------------

    def _clamp(self, v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(v)))

    def _clamp_all(self) -> None:
        self.state.arousal = self._clamp(self.state.arousal, self.cfg.arousal_min, self.cfg.arousal_max)
        self.state.confidence = self._clamp(self.state.confidence, self.cfg.confidence_min, self.cfg.confidence_max)
        self.state.confidence_floor = self._clamp(self.state.confidence_floor, 0.0, self.cfg.confidence_floor_max)
        self.state.uncertainty = self._clamp(self.state.uncertainty, self.cfg.uncertainty_min, self.cfg.uncertainty_max)
        self.state.curiosity = self._clamp(self.state.curiosity, self.cfg.curiosity_min, self.cfg.curiosity_max)
        self.state.valence = self._clamp(self.state.valence, self.cfg.valence_min, self.cfg.valence_max)

    def _snapshot(self) -> Dict[str, float]:
        return {
            "arousal": float(self.state.arousal),
            "valence": float(self.state.valence),
            "confidence": float(self.state.confidence),
            "confidence_floor": float(self.state.confidence_floor),
            "uncertainty": float(self.state.uncertainty),
            "curiosity": float(self.state.curiosity),
        }

    def _delta(self, prev: Dict[str, float]) -> Dict[str, float]:
        cur = self._snapshot()
        return {k: float(cur[k] - prev.get(k, 0.0)) for k in cur.keys()}
