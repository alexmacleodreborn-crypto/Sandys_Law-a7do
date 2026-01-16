import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from bootstrap.system import SystemBootstrap


# ============================================================
# Streamlit Setup
# ============================================================

st.set_page_config(page_title="A7DO — World", layout="wide")
st.title("A7DO — Visual World")

if "system" not in st.session_state:
    st.session_state.system = SystemBootstrap(enable_autonomy=False)

system = st.session_state.system


# ============================================================
# Controls
# ============================================================

st.sidebar.header("Controls")

col1, col2, col3 = st.sidebar.columns(3)
if col2.button("⬆️"):
    result = system.apply_move(0, -1)
elif col1.button("⬅️"):
    result = system.apply_move(-1, 0)
elif col3.button("➡️"):
    result = system.apply_move(1, 0)
elif st.sidebar.button("⬇️"):
    result = system.apply_move(0, 1)
elif st.sidebar.button("Wait"):
    result = system.step()
else:
    result = None

auto = st.sidebar.checkbox("Enable Autonomy", value=system.enable_autonomy)
system.enable_autonomy = auto


# ============================================================
# World Rendering
# ============================================================

st.subheader("World State")

world = system.world.snapshot()
size_x, size_y = world["size"]
agent_pos = world["agent"]

grid = np.zeros((size_y, size_x))

if agent_pos:
    ax, ay = agent_pos
    grid[ay, ax] = 1.0

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(grid, origin="upper")
ax.set_xticks(range(size_x))
ax.set_yticks(range(size_y))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True)

st.pyplot(fig)


# ============================================================
# Latest Events
# ============================================================

st.subheader("Recent Events")
recent = system.memory.recent(10)
for e in recent:
    st.text(e.summary())
