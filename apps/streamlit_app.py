# ============================================================
# Streamlit entrypoint for A7DO
# ============================================================

import sys
from pathlib import Path

# ------------------------------------------------------------
# Ensure project root is on PYTHONPATH (Streamlit requirement)
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bootstrap.system import SystemBootstrap
from shared.events import EventType


# ============================================================
# Streamlit Setup
# ============================================================

st.set_page_config(
    page_title="A7DO Control Room",
    layout="wide",
)

st.title("A7DO — Living Cognitive System")

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------

if "system" not in st.session_state:
    st.session_state.system = SystemBootstrap(enable_autonomy=False)

system: SystemBootstrap = st.session_state.system


# ============================================================
# Sidebar Controls
# ============================================================

st.sidebar.header("Controls")

system.enable_autonomy = st.sidebar.checkbox(
    "Enable Autonomy",
    value=system.enable_autonomy,
)

if st.sidebar.button("Background Step"):
    system.step()

if st.sidebar.button("Shutdown System"):
    system.close()
    st.stop()


# ============================================================
# Tabs
# ============================================================

tab_world, tab_health, tab_memory = st.tabs(
    ["🌍 World", "📈 Health & Phase", "🧠 Memory"]
)

# ============================================================
# 🌍 WORLD TAB
# ============================================================

with tab_world:
    st.subheader("World")

    col_left, col_mid, col_right = st.columns([1, 2, 1])

    with col_left:
        if st.button("⬆️"):
            system.apply_move(0, -1)
        if st.button("⬇️"):
            system.apply_move(0, 1)

    with col_right:
        if st.button("⬅️"):
            system.apply_move(-1, 0)
        if st.button("➡️"):
            system.apply_move(1, 0)

    world = system.world.snapshot()
    size_x, size_y = world["size"]
    agent = world.get("agent")

    grid = np.zeros((size_y, size_x))
    if agent:
        x, y = agent
        grid[y, x] = 1.0

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(grid, origin="upper")
    ax.set_xticks(range(size_x))
    ax.set_yticks(range(size_y))
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    st.pyplot(fig)


# ============================================================
# 📈 HEALTH & PHASE TAB
# ============================================================

with tab_health:
    st.subheader("Internal Regulation")

    # Pull recent memory
    recent_events = system.memory.recent(300)

    # --------------------------------------------------------
    # Build dataframe from INTERNAL events
    # --------------------------------------------------------

    rows = []
    for e in recent_events:
        if (
            e.type == EventType.INTERNAL
            and e.name in ("state_update", "light_tick")
        ):
            state = e.payload.get("state", {})
            rows.append(
                {
                    "arousal": state.get("arousal"),
                    "confidence": state.get("confidence"),
                    "confidence_floor": state.get("confidence_floor"),
                    "uncertainty": state.get("uncertainty"),
                    "curiosity": state.get("curiosity"),
                }
            )

    df = pd.DataFrame(rows)

    if not df.empty:
        st.line_chart(
            df[
                [
                    "arousal",
                    "confidence",
                    "confidence_floor",
                    "uncertainty",
                    "curiosity",
                ]
            ]
        )
    else:
        st.info("No internal state data yet — system is stabilising.")

    # --------------------------------------------------------
    # Health + Phase snapshots
    # --------------------------------------------------------

    snapshot = system.snapshot()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Health Snapshot")
        st.json(snapshot["health"])

    with col2:
        st.subheader("Phase Snapshot")
        st.json(snapshot["phase"])


# ============================================================
# 🧠 MEMORY TAB
# ============================================================

with tab_memory:
    st.subheader("Recent Memory Events")

    for e in system.memory.recent(40):
        st.text(e.summary())
