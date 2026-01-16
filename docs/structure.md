# Sandy’s Law — System Structure (Authoritative)

This document defines the fixed directory and responsibility structure
for the A7DO + World + SLED unified system.

No module may violate these boundaries.

---

## Root

sandyslaw-a7do/
- Repository root and governance

---

## a7do/ — Cognitive Organism

Responsible for subjective experience, learning, memory, identity,
emotion, and decision-making.

No direct access to world truth.

### a7do/core/
- Agent container
- Event loop
- Internal registries
- No domain logic

### a7do/background_core/
- Always-on regulation
- Memory consolidation
- Internal pressure & motivation
- Runs even when idle

### a7do/embodiment/
- Body state
- Senses (vision, touch, audio, proprioception)
- Actuators (movement, manipulation)

### a7do/cognition/
- Perception
- Memory (episodic, semantic, procedural)
- Self-model (identity, emotions, needs)
- Reasoning & planning
- Language grounding & dialogue

### a7do/policies/
- Safety constraints
- Permission boundaries
- Reward shaping

---

## world/ — Objective Reality

World contains no cognition, emotion, or intent.

### world/grid/
- Spatial layout
- Navigation topology

### world/physics/
- Collision
- Interaction rules
- Visibility

### world/objects/
- Rooms, doors, items
- No semantics

### world/scenarios/
- YAML-defined environments

---

## sled/ — Entropy & Decision Engine

Reusable across finance, cognition, and systems.

### sled/square/
- Z–Σ phase system
- Persistence & clustering

### sled/signals/
- Event detection
- Phase transitions

### sled/risk/
- Stability limits
- Zeno / expiry detection

---

## shared/ — Integration Layer (Critical)

Only place where systems touch.

### shared/events.py
- Unified event schema

### shared/memory.py
- Storage & retrieval contracts

### shared/schemas.py
- Typed data models

### shared/health.py
- System stability & burnout prevention

---

## apps/ — Runners & Interfaces

No logic lives here.

### apps/chat_cli.py
- Terminal interaction

### apps/sim_runner.py
- Headless simulation loop

### apps/streamlit/
- UI only (later)

---

## data/ — Runtime State (Ignored by Git)

- Memory databases
- Logs
- Snapshots

---

## docs/
- Architecture
- Genesis
- Ethics
- Milestones

---

## tests/
- Unit tests only
- No experiments

---

## Rule

If a file does not clearly belong in one folder, the design is wrong.
