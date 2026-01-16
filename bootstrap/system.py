from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from a7do.background_core.core import BackgroundCore
from a7do.perception.perception import Perception, PerceptionEngine
from a7do.core.agent import A7DOAgent
from a7do.state.identity import IdentityStore, IdentityRecord

from shared.events import Event, action, observation, system_event
from shared.memory import MemoryStore
from shared.health import HealthAnalyzer, HealthSnapshot

from world.world import World
from sled.phase.core import PhaseAnalyzer, PhaseSnapshot


# ============================================================
# Bootstrap Output Bundle
# ============================================================

@dataclass(frozen=True)
class StepResult:
    """
    Result of one orchestrated system step.
    """
    emitted_events: List[Event]
    percepts: List[Perception]
    health: HealthSnapshot
    phase: PhaseSnapshot
    identity: IdentityRecord
    world_snapshot: Dict[str, Any]


# ============================================================
# System Bootstrap (Coupling Only)
# ============================================================

class SystemBootstrap:
    """
    Coupling layer only.

    This class wires together otherwise independent modules.
    It must contain no domain logic beyond ordering, routing, and plumbing.
    """

    def __init__(
        self,
        *,
        memory_path: str = "data/memory/memory.db",
        identity_path: str = "data/identity/identity.json",
        world_size: Tuple[int, int] = (5, 5),
        spawn_at: Tuple[int, int] = (2, 2),
        enable_autonomy: bool = False,
    ) -> None:
        self.identity_store = IdentityStore(path=identity_path)
        self.identity = self.identity_store.get()

        self.memory = MemoryStore(db_path=memory_path)

        self.world = World(width=world_size[0], height=world_size[1])

        # Tools
        self.bg = BackgroundCore(self.memory)
        self.perception = PerceptionEngine()
        self.health = HealthAnalyzer()
        self.phase = PhaseAnalyzer()

        # Optional agent
        self.enable_autonomy = bool(enable_autonomy)
        self.agent = A7DOAgent(self.memory)

        # Boot world and record initial event
        boot_events = self.world.spawn_agent(spawn_at[0], spawn_at[1])
        self.memory.append_many(boot_events)

        # Emit system boot event (for traceability)
        self.memory.append(
            system_event(
                source="bootstrap",
                name="boot",
                payload={
                    "identity_id": self.identity.identity_id,
                    "genesis_id": self.identity.genesis_id,
                    "incarnation": self.identity.incarnation,
                },
            )
        )

    # ----------------------------
    # Lifecycle
    # ----------------------------

    def close(self) -> None:
        self.memory.close()

    # ----------------------------
    # External interaction
    # ----------------------------

    def apply_move(self, dx: int, dy: int, source: str = "user") -> StepResult:
        """
        Manual movement injection. This is the simplest possible action.
        """
        act = action(source=source, name="move", payload={"dx": int(dx), "dy": int(dy)})
        self.memory.append(act)

        world_events = self.world.step(act)
        self.memory.append_many(world_events)

        # Background core always runs
        internal_events = self.bg.step()

        # Agent observes outcomes (bias update), but does not act unless enabled
        self.agent.observe_outcomes()

        emitted = world_events + internal_events
        percepts = self.perception.process(emitted)

        return self._bundle_result(emitted, percepts)

    def step(self, user_text: Optional[str] = None) -> StepResult:
        """
        One orchestrated step.

        If autonomy is enabled, A7DO may choose an action.
        Otherwise this performs a background-only cycle.
        """
        emitted: List[Event] = []

        # Treat user text as a raw observation signal (no semantics here).
        if user_text:
            e = observation(source="user", name="utterance", payload={"text": str(user_text)})
            self.memory.append(e)
            emitted.append(e)

        # Background core always runs
        internal_events = self.bg.step()
        emitted.extend(internal_events)

        # Agent updates bias from recent outcomes
        self.agent.observe_outcomes()

        # Optional autonomy: agent decides a move
        if self.enable_autonomy:
            act = self.agent.decide()
            if act is not None:
                self.memory.append(act)
                emitted.append(act)

                world_events = self.world.step(act)
                self.memory.append_many(world_events)
                emitted.extend(world_events)

                # Run background core after acting
                internal_events2 = self.bg.step()
                emitted.extend(internal_events2)

        percepts = self.perception.process(emitted)
        return self._bundle_result(emitted, percepts)

    # ----------------------------
    # Bundling & snapshots
    # ----------------------------

    def _bundle_result(self, emitted: List[Event], percepts: List[Perception]) -> StepResult:
        # Analyze health/phase over recent window only (advisory)
        recent_events = self.memory.recent(200)
        health = self.health.analyze(recent_events)
        phase = self.phase.analyze(recent_events)

        return StepResult(
            emitted_events=emitted,
            percepts=percepts,
            health=health,
            phase=phase,
            identity=self.identity,
            world_snapshot=self.world.snapshot(),
        )

    def snapshot(self) -> Dict[str, Any]:
        """
        UI-friendly snapshot (read-only).
        """
        recent = self.memory.recent(50)
        health = self.health.analyze(recent)
        phase = self.phase.analyze(recent)

        return {
            "identity": {
                "identity_id": self.identity.identity_id,
                "genesis_id": self.identity.genesis_id,
                "incarnation": self.identity.incarnation,
                "continuity_version": self.identity.continuity_version,
            },
            "world": self.world.snapshot(),
            "health": {
                "event_count": health.event_count,
                "arousal": health.arousal,
                "confidence": health.confidence,
                "confidence_floor": health.confidence_floor,
                "uncertainty": health.uncertainty,
                "curiosity": health.curiosity,
                "zeno_risk": health.zeno_risk,
                "burnout_risk": health.burnout_risk,
                "stagnation_risk": health.stagnation_risk,
                "notes": health.notes,
            },
            "phase": {
                "event_count": phase.event_count,
                "internal_density": phase.internal_density,
                "action_density": phase.action_density,
                "observation_density": phase.observation_density,
                "clustering": phase.clustering,
                "volatility": phase.volatility,
                "coherence": phase.coherence,
                "phase_state": phase.phase_state,
                "notes": phase.notes,
            },
            "recent_events": [e.summary() for e in recent],
        }
