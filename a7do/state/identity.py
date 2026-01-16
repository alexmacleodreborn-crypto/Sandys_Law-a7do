from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
from uuid import uuid4


# ============================================================
# Identity Record (Minimal & Persistent)
# ============================================================

@dataclass(frozen=True)
class IdentityRecord:
    """
    Minimal persistent identity for A7DO.

    This is NOT personality.
    This is NOT cognition.
    This is the continuity anchor across restarts.
    """

    identity_id: str
    genesis_id: str
    creation_tag: str

    incarnation: int          # increments on controlled rebuild
    continuity_version: int   # increments only if doctrine changes

    notes: Optional[str] = None


# ============================================================
# Identity Store
# ============================================================

class IdentityStore:
    """
    Persistent identity manager.

    Doctrine:
    - Identity is created once.
    - Identity is loaded, not regenerated.
    - Destruction must be explicit and logged.
    """

    def __init__(
        self,
        path: str = "data/identity/identity.json",
        continuity_version: int = 1,
        creation_tag: str = "a7do",
    ) -> None:
        self.path = path
        self.continuity_version = int(continuity_version)
        self.creation_tag = creation_tag

        self._identity: Optional[IdentityRecord] = None

        self._ensure_dir()
        self._load_or_create()

    # ----------------------------
    # Public API
    # ----------------------------

    def get(self) -> IdentityRecord:
        if not self._identity:
            raise RuntimeError("Identity not initialized.")
        return self._identity

    def exists(self) -> bool:
        return self._identity is not None

    # ----------------------------
    # Controlled lifecycle
    # ----------------------------

    def rebuild_incarnation(self, note: Optional[str] = None) -> IdentityRecord:
        """
        Controlled rebuild of the system body while preserving identity.

        Increments incarnation.
        Does NOT change identity_id or genesis_id.
        """
        if not self._identity:
            raise RuntimeError("Cannot rebuild non-existent identity.")

        rec = self._identity
        new_rec = IdentityRecord(
            identity_id=rec.identity_id,
            genesis_id=rec.genesis_id,
            creation_tag=rec.creation_tag,
            incarnation=rec.incarnation + 1,
            continuity_version=self.continuity_version,
            notes=note,
        )

        self._write(new_rec)
        self._identity = new_rec
        return new_rec

    def destroy_identity(self, confirm: bool) -> None:
        """
        Explicit identity destruction.

        This is a terminal operation.
        """
        if not confirm:
            raise RuntimeError(
                "Identity destruction requires explicit confirmation."
            )

        if os.path.exists(self.path):
            os.remove(self.path)

        self._identity = None

    # ----------------------------
    # Internal
    # ----------------------------

    def _ensure_dir(self) -> None:
        d = os.path.dirname(os.path.abspath(self.path))
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    def _load_or_create(self) -> None:
        if os.path.exists(self.path):
            self._identity = self._read()
            return

        # Create new identity (Genesis moment)
        genesis_id = str(uuid4())
        identity_id = str(uuid4())

        rec = IdentityRecord(
            identity_id=identity_id,
            genesis_id=genesis_id,
            creation_tag=self.creation_tag,
            incarnation=1,
            continuity_version=self.continuity_version,
            notes="genesis",
        )

        self._write(rec)
        self._identity = rec

    def _read(self) -> IdentityRecord:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return IdentityRecord(
            identity_id=data["identity_id"],
            genesis_id=data["genesis_id"],
            creation_tag=data["creation_tag"],
            incarnation=int(data["incarnation"]),
            continuity_version=int(data["continuity_version"]),
            notes=data.get("notes"),
        )

    def _write(self, rec: IdentityRecord) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(rec), f, indent=2)
        os.replace(tmp, self.path)
