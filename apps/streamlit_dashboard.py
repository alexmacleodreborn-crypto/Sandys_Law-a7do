import streamlit as st
import pandas as pd

from bootstrap.system import SystemBootstrap


# ============================================================
# Setup
# ============================================================

st.set_page_config(page_title="A7DO — Dashboard", layout="wide")
st.title("A7DO — Health & Phase Dashboard")

if "system" not in st.session_state:
    st.session_state.system = SystemBootstrap(enable_autonomy=True)

system = st.session_state.system


# ============================================================
# Step System
# ============================================================

if st.button("Advance One Step"):
    system.step()

recent = system.memory.recent(100)


# ============================================================
# Extract State History
# ============================================================

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


# ============================================================
# Plots
# ============================================================

st.subheader("Internal State Over Time")

if not df.empty:
    st.line_chart(
        df[["arousal", "confidence", "confidence_floor", "uncertainty", "curiosity"]]
    )
else:
    st.info("No internal state data yet.")


# ============================================================
# Health & Phase Snapshot
# ============================================================

snap = system.snapshot()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Health")
    st.json(snap["health"])

with col2:
    st.subheader("Phase")
    st.json(snap["phase"])
