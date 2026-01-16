from __future__ import annotations

import sys
from typing import List

from shared.events import action
from shared.memory import MemoryStore
from a7do.background_core.core import BackgroundCore
from world.world import World


# ============================================================
# Helpers
# ============================================================

def print_events(events: List):
    for e in events:
        print(f"  • {e.summary()} | payload={e.payload}")


def print_state(internal_events: List):
    for e in internal_events:
        if e.name in ("state_update", "light_tick"):
            state = e.payload.get("state", {})
            print(
                f"    [STATE] arousal={state.get('arousal'):.2f} "
                f"conf={state.get('confidence'):.2f} "
                f"floor={state.get('confidence_floor'):.2f} "
                f"uncert={state.get('uncertainty'):.2f} "
                f"curiosity={state.get('curiosity'):.2f}"
            )


# ============================================================
# Main Loop
# ============================================================

def main():
    print("=== A7DO CLI ===")
    print("Commands:")
    print("  w/a/s/d  -> move")
    print("  wait     -> no action (observe background core)")
    print("  state    -> print last internal state")
    print("  quit     -> exit")
    print()

    # ----------------------------
    # Boot system
    # ----------------------------

    with MemoryStore() as memory:
        bg = BackgroundCore(memory)

        world = World(width=5, height=5)
        world_events = world.spawn_agent(2, 2)
        memory.append_many(world_events)

        print("World initialized.")
        print_events(world_events)

        last_internal_events: List = []

        # ----------------------------
        # Interactive loop
        # ----------------------------

        while True:
            try:
                cmd = input("\n> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")

