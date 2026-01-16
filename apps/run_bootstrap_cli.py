from __future__ import annotations

import sys
from typing import Optional

from bootstrap.system import SystemBootstrap


# ============================================================
# CLI Utilities
# ============================================================

def print_help():
    print(
        """
Commands:
  w/a/s/d     -> move (manual)
  wait        -> background-only step
  auto on     -> enable A7DO autonomy
  auto off    -> disable A7DO autonomy
  state       -> show system snapshot
  help        -> show this help
  quit        -> exit
"""
    )


def print_step(result):
    if result.emitted_events:
        print("\n[EVENTS]")
        for e in result.emitted_events:
            print(" •", e.summary())

    if result.percepts:
        print("\n[PERCEPTS]")
        for p in result.percepts:
            print(f" • {p.type}: {p.payload}")

    print("\n[HEALTH]")
    h = result.health
    print(
        f" events={h.event_count} "
        f"arousal={h.arousal:.2f if h.arousal is not None else 'n/a'} "
        f"conf={h.confidence:.2f if h.confidence is not None else 'n/a'} "
        f"floor={h.confidence_floor:.2f if h.confidence_floor is not None else 'n/a'} "
        f"uncert={h.uncertainty:.2f if h.uncertainty is not None else 'n/a'} "
        f"curiosity={h.curiosity:.2f if h.curiosity is not None else 'n/a'}"
    )
    print(
        f" risks: zeno={h.zeno_risk:.2f} "
        f"burnout={h.burnout_risk:.2f} "
        f"stagnation={h.stagnation_risk:.2f} "
        f"notes={h.notes}"
    )

    print("\n[PHASE]")
    p = result.phase
    print(
        f" state={p.phase_state} "
        f"coherence={p.coherence:.2f} "
        f"volatility={p.volatility:.2f} "
        f"clustering={p.clustering:.2f} "
        f"notes={p.notes}"
    )

    print("\n[WORLD]")
    print(" ", result.world_snapshot)


# ============================================================
# Main Runner
# ============================================================

def main():
    print("=== A7DO Bootstrap Runner ===")
    print_help()

    system = SystemBootstrap(enable_autonomy=False)

    try:
        while True:
            try:
                cmd = input("\n> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not cmd:
                continue

            if cmd in ("quit", "exit"):
                break

            if cmd == "help":
                print_help()
                continue

            if cmd == "state":
                snap = system.snapshot()
                print("\n[SNAPSHOT]")
                for k, v in snap.items():
                    print(f"{k}: {v}")
                continue

            if cmd == "wait":
                result = system.step()
                print_step(result)
                continue

            if cmd == "auto on":
                system.enable_autonomy = True
                print("Autonomy enabled.")
                continue

            if cmd == "auto off":
                system.enable_autonomy = False
                print("Autonomy disabled.")
                continue

            if cmd in ("w", "a", "s", "d"):
                dx, dy = 0, 0
                if cmd == "w":
                    dy = -1
                elif cmd == "s":
                    dy = 1
                elif cmd == "a":
                    dx = -1
                elif cmd == "d":
                    dx = 1

                result = system.apply_move(dx, dy)
                print_step(result)
                continue

            # Any other text becomes a raw user observation
            result = system.step(user_text=cmd)
            print_step(result)

    finally:
        system.close()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
