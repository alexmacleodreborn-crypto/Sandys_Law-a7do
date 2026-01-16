import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH (required for Streamlit)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bootstrap.system import SystemBootstrap



# ============================================================
# Setup
# ============================================================

st.set_page_config(
    page_title="A7DO Control Room",
    layout="wide",
)

st.title("A7DO — Living System Control Room")

if "system" not in st.session_state:
    st.session_state.system = SystemBootstrap(enable_autonomy=False)

system = st.session_state.system


# ============================================================
# Sidebar Controls (Global)
# ============================================================

st.sidebar.header("Global Controls")

system.enable_autonomy = st.sidebar.checkbox(
    "Enable Autonomy", value=system.enable_autonomy
)

if st.sidebar.button("Background Step"):
    system.step()

if st.sidebar.button("Hard Refresh (UI only)"):
    st.rerun()


# ============================================================
# Tabs
# ============================================================

tab_world, tab_health, tab_memory, tab_settings = st.tabs(
    ["🌍 World", "📈 Health & Phase", "🧠 Memory", "⚙️ Settings"]
)

# ------------------------------------------------------------
# 🌍 WORLD TAB
# ------------------------------------------------------------

with tab_world:
    st.subheader("World")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬆️"):
            system.apply_move(0, -1)
        if st.button("⬇️"):
            system.apply_move(0, 1)

    with col3:
        if st.button("⬅️"):
            system.apply_move(-1, 0)
        if st.button("➡️"):
            system.apply_move(1, 0)

    world = system.world.snapshot()
    size_x, size_y = world["size"]
    agent = world["agent"]

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

# ------------------------------------------------------------
# 📈 HEALTH & PHASE TAB
# ------------------------------------------------------------

with tab_health:
    st.subheader("Internal State Over Time")

    recent = system.memory.recent(200)

    rows = []
    for e in recent:
        if e.type.value == "internal" and e.name == "state_update":
            s = e.payload.get("state", {})
            rows.append(
                {
                    "arousal": s.get("arousal"),
                    "confidence": s.get("confidence"),
                    "confidence_floor": s.get("confidence_floor"),
                    "uncertainty": s.get("uncertainty"),
                    "curiosity": s.get("curiosity"),
                }
            )

    df = pd.DataFrame(rows)

    if not df.empty:
        st.line_chart(
            df[["arousal", "confidence", "confidence_floor", "uncertainty", "curiosity"]]
        )
    else:
        st.info("No internal state data yet.")

    snap = system.snapshot()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Health")
        st.json(snap["health"])

    with col2:
        st.subheader("Phase")
        st.json(snap["phase"])

# ------------------------------------------------------------
# 🧠 MEMORY TAB
# ------------------------------------------------------------

with tab_memory:
    st.subheader("Recent Events")

    for e in system.memory.recent(25):
        st.text(e.summary())

# ------------------------------------------------------------
# ⚙️ SETTINGS TAB
# ------------------------------------------------------------

with tab_settings:
    st.subheader("System Identity")

    st.json(
        {
            "identity_id": system.identity.identity_id,
            "genesis_id": system.identity.genesis_id,
            "incarnation": system.identity.incarnation,
            "continuity_version": system.identity.continuity_version,
        }
    )

    st.warning(
        "Core modules are frozen. "
        "Only bootstrap and UI layers may change."
    )
