from __future__ import annotations

import sys
from pathlib import Path

# ------------------------------------------------------------
# Ensure PROJECT ROOT is on PYTHONPATH
# bootstrap/system.py → parents[2] = repo root
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------
# Imports (now safe)
# ------------------------------------------------------------

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from a7do.background_core.core import BackgroundCore
from a7do.perception.perception import Percept, PerceptionEngine
from a7do.core.agent import A7DOAgent
from a7do.state.identity import IdentityStore, IdentityRecord
from a7do.cognition.prediction import PredictionEngine

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
# System Bootstrap (Coupling / Ordering Only)
# ============================================================

class SystemBootstrap:
    """
    Coupling layer only.

    Responsibilities:
    - Lifecycle ownership
    - Ordering guarantees
    - Module wiring

    Contains NO domain logic.
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

        # ----------------------------
        # Identity
        # ----------------------------
        self.identity_store = IdentityStore(path=identity_path)
        self.identity = self.identity_store.get()

        # ----------------------------
        # Memory
        # ----------------------------
        self.memory = MemoryStore(db_path=memory_path)
        self.memory.__enter__()

        # ----------------------------
        # World
        # ----------------------------
        self.world = World(width=world_size[0], height=world_size[1])

        # ----------------------------
        # Engines
        # ----------------------------
        self.bg = BackgroundCore(self.memory)
        self.perception = PerceptionEngine()
        self.health = HealthAnalyzer()
        self.phase = PhaseAnalyzer()
        self.predictor = PredictionEngine()

        # ----------------------------
        # Agent
        # ----------------------------
        self.enable_autonomy = bool(enable_autonomy)
        self.agent = A7DOAgent(self.memory)

        # ----------------------------
        # Spawn + boot
        # ----------------------------
        spawn_events = self.world.spawn_agent(spawn_at[0], spawn_at[1])
        self.memory.append_many(spawn_events)

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

        # Initial regulation tick
        self.bg.step()

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def close(self) -> None:
        try:
            self.memory.__exit__(None, None, None)
        except Exception:
            pass

    # --------------------------------------------------------
    # External control
    # --------------------------------------------------------

    def apply_move(self, dx: int, dy: int, source: str = "user") -> StepResult:
        emitted: List[Event] = []

        # Action
        act = action(source=source, name="move", payload={"dx": dx, "dy": dy})
        self.memory.append(act)
        emitted.append(act)

        # World step
        world_events = self.world.step(act)
        self.memory.append_many(world_events)
        emitted.extend(world_events)

        # Prediction error
        pred_events = self.predictor.observe(emitted)
        if pred_events:
            self.memory.append_many(pred_events)
            emitted.extend(pred_events)

        # Background regulation
        emitted.extend(self.bg.step())

        self.agent.observe_outcomes()
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

                pred_events = self.predictor.observe(emitted)
                if pred_events:
                    self.memory.append_many(pred_events)
                    emitted.extend(pred_events)

                emitted.extend(self.bg.step())

        percepts = self.perception.process(emitted)
        return self._bundle(emitted, percepts)

    # --------------------------------------------------------
    # Bundling
    # --------------------------------------------------------

    def _bundle(self, emitted: List[Event], percepts: List[Percept]) -> StepResult:
        recent = self.memory.recent(300)
        return StepResult(
            emitted_events=emitted,
            percepts=percepts,
            health=self.health.analyze(recent),
            phase=self.phase.analyze(recent),
            identity=self.identity,
            world_snapshot=self.world.snapshot(),
        )

    def snapshot(self) -> Dict[str, Any]:
        recent = self.memory.recent(300)
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
