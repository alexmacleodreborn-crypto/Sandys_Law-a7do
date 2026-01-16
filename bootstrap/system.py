from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from a7do.background_core.core import BackgroundCore
from a7do.perception.perception import Percept, PerceptionEngine
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
    emitted_events: List[Event]
    percepts: List[Percept]
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
    Responsible for lifecycle wiring and ordering.
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

        # Identity
        self.identity_store = IdentityStore(path=identity_path)
        self.identity = self.identity_store.get()

        # Memory (explicit lifecycle)
        self.memory = MemoryStore(db_path=memory_path)
        self.memory.open()

        # World
        self.world = World(width=world_size[0], height=world_size[1])

        # Engines
        self.bg = BackgroundCore(self.memory)
        self.perception = PerceptionEngine()
        self.health = HealthAnalyzer()
        self.phase = PhaseAnalyzer()

        # Agent
        self.enable_autonomy = bool(enable_autonomy)
        self.agent = A7DOAgent(self.memory)

        # Spawn agent
        spawn_events = self.world.spawn_agent(spawn_at[0], spawn_at[1])
        self.memory.append_many(spawn_events)

        # Boot event
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

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def close(self) -> None:
        if self.memory:
            self.memory.close()

    # --------------------------------------------------------
    # External control
    # --------------------------------------------------------

    def apply_move(self, dx: int, dy: int, source: str = "user") -> StepResult:
        act = action(source=source, name="move", payload={"dx": dx, "dy": dy})
        self.memory.append(act)

        world_events = self.world.step(act)
        self.memory.append_many(world_events)

        internal_events = self.bg.step()
        self.agent.observe_outcomes()

        emitted = world_events + internal_events
        percepts = self.perception.process(emitted)

        return self._bundle(emitted, percepts)

    def step(self, user_text: Optional[str] = None) -> StepResult:
        emitted: List[Event] = []

        if user_text:
            e = observation(
                source="user",
                name="utterance",
                payload={"text": str(user_text)},
            )
            self.memory.append(e)
            emitted.append(e)

        emitted.extend(self.bg.step())
        self.agent.observe_outcomes()

        if self.enable_autonomy:
            act = self.agent.decide()
            if act:
                self.memory.append(act)
                emitted.append(act)

                world_events = self.world.step(act)
                self.memory.append_many(world_events)
                emitted.extend(world_events)
                emitted.extend(self.bg.step())

        percepts = self.perception.process(emitted)
        return self._bundle(emitted, percepts)

    # --------------------------------------------------------
    # Bundling
    # --------------------------------------------------------

    def _bundle(self, emitted: List[Event], percepts: List[Percept]) -> StepResult:
        recent = self.memory.recent(200)
        return StepResult(
            emitted_events=emitted,
            percepts=percepts,
            health=self.health.analyze(recent),
            phase=self.phase.analyze(recent),
            identity=self.identity,
            world_snapshot=self.world.snapshot(),
        )

    def snapshot(self) -> Dict[str, Any]:
        recent = self.memory.recent(200)
        return {
            "identity": {
                "identity_id": self.identity.identity_id,
                "genesis_id": self.identity.genesis_id,
                "incarnation": self.identity.incarnation,
                "continuity_version": self.identity.continuity_version,
            },
            "world": self.world.snapshot(),
            "health": self.health.analyze(recent).__dict__,
            "phase": self.phase.analyze(recent).__dict__,
        }
