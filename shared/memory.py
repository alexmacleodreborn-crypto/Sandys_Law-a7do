from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Tuple

from shared.events import Event, EventType


# ============================================================
# Storage Schema (Event-based, not time-based)
# ============================================================

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,

    parent_id TEXT NULL,

    type TEXT NOT NULL,
    source TEXT NOT NULL,
    name TEXT NOT NULL,

    payload_json TEXT NOT NULL,
    confidence REAL NULL
);

CREATE INDEX IF NOT EXISTS idx_events_parent_id ON events(parent_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);
CREATE INDEX IF NOT EXISTS idx_events_name ON events(name);
"""


# ============================================================
# Exceptions
# ============================================================

class MemoryError(Exception):
    pass


class MemoryIntegrityError(MemoryError):
    pass


# ============================================================
# Helpers
# ============================================================

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _event_to_row(e: Event) -> Dict[str, Any]:
    try:
        payload_json = json.dumps(e.payload, ensure_ascii=False, separators=(",", ":"))
    except TypeError as ex:
        raise MemoryError(
            f"Event payload is not JSON serializable for event {e.id} ({e.name})."
        ) from ex

    return {
        "id": e.id,
        "parent_id": e.parent_id,
        "type": e.type.value if isinstance(e.type, EventType) else str(e.type),
        "source": e.source,
        "name": e.name,
        "payload_json": payload_json,
        "confidence": e.confidence,
    }


def _row_to_event(row: sqlite3.Row) -> Event:
    payload = json.loads(row["payload_json"]) if row["payload_json"] else {}
    return Event(
        type=EventType(row["type"]),
        source=row["source"],
        name=row["name"],
        payload=payload,
        id=row["id"],
        parent_id=row["parent_id"],
        confidence=row["confidence"],
    )


# ============================================================
# Memory Store
# ============================================================

class MemoryStore:
    """
    Persistent append-only event store (SQLite).

    Doctrine:
    - Append-only
    - No time dependence
    - Ordering via seq only
    """

    def __init__(self, db_path: str = "data/memory/memory.db") -> None:
        _ensure_dir(db_path)
        self.db_path = db_path

        # Streamlit / UI apps use multiple threads.
        # SQLite defaults to single-thread usage, so we explicitly allow cross-thread access.
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    # ----------------------------
    # Lifecycle
    # ----------------------------

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ----------------------------
    # Writes
    # ----------------------------

    def append(self, event: Event) -> int:
        row = _event_to_row(event)
        try:
            cur = self._conn.execute(
                """
                INSERT INTO events (id, parent_id, type, source, name, payload_json, confidence)
                VALUES (:id, :parent_id, :type, :source, :name, :payload_json, :confidence)
                """,
                row,
            )
            self._conn.commit()
            return int(cur.lastrowid)
        except sqlite3.IntegrityError as ex:
            raise MemoryIntegrityError(
                f"Integrity error while appending event {event.id} ({event.name})."
            ) from ex

    def append_many(self, events: Iterable[Event]) -> Tuple[int, int]:
        events_list = list(events)
        if not events_list:
            return (0, self.last_seq())

        rows = [_event_to_row(e) for e in events_list]
        try:
            self._conn.executemany(
                """
                INSERT INTO events (id, parent_id, type, source, name, payload_json, confidence)
                VALUES (:id, :parent_id, :type, :source, :name, :payload_json, :confidence)
                """,
                rows,
            )
            self._conn.commit()
        except sqlite3.IntegrityError as ex:
            raise MemoryIntegrityError(
                "Integrity error while appending multiple events."
            ) from ex

        return (len(events_list), self.last_seq())

    # ----------------------------
    # Reads
    # ----------------------------

    def get(self, event_id: str) -> Optional[Event]:
        cur = self._conn.execute(
            "SELECT * FROM events WHERE id = ? LIMIT 1",
            (event_id,),
        )
        row = cur.fetchone()
        return _row_to_event(row) if row else None

    def get_with_seq(self, event_id: str) -> Optional[Tuple[int, Event]]:
        cur = self._conn.execute(
            "SELECT * FROM events WHERE id = ? LIMIT 1",
            (event_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return (int(row["seq"]), _row_to_event(row))

    def last_seq(self) -> int:
        cur = self._conn.execute("SELECT COALESCE(MAX(seq), 0) AS m FROM events")
        return int(cur.fetchone()["m"])

    def recent(self, n: int = 50) -> List[Event]:
        n = max(0, int(n))
        cur = self._conn.execute(
            "SELECT * FROM events ORDER BY seq DESC LIMIT ?",
            (n,),
        )
        rows = cur.fetchall()
        return [_row_to_event(r) for r in reversed(rows)]

    def iter_since(self, seq_exclusive: int) -> Iterable[Tuple[int, Event]]:
        cur = self._conn.execute(
            "SELECT * FROM events WHERE seq > ? ORDER BY seq ASC",
            (int(seq_exclusive),),
        )
        for row in cur:
            yield (int(row["seq"]), _row_to_event(row))

    def find(
        self,
        *,
        type: Optional[EventType] = None,
        source: Optional[str] = None,
        name: Optional[str] = None,
        parent_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[Tuple[int, Event]]:
        clauses = []
        params: List[Any] = []

        if type is not None:
            clauses.append("type = ?")
            params.append(type.value)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if name is not None:
            clauses.append("name = ?")
            params.append(name)
        if parent_id is not None:
            clauses.append("parent_id = ?")
            params.append(parent_id)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        limit = max(1, int(limit))

        cur = self._conn.execute(
            f"SELECT * FROM events {where} ORDER BY seq ASC LIMIT ?",
            (*params, limit),
        )
        rows = cur.fetchall()
        return [(int(r["seq"]), _row_to_event(r)) for r in rows]

    # ----------------------------
    # Stats / Auditing
    # ----------------------------

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) AS c FROM events")
        return int(cur.fetchone()["c"])

    def stats(self) -> Dict[str, Any]:
        cur = self._conn.execute(
            """
            SELECT
              COUNT(*) AS total,
              SUM(CASE WHEN type = 'observation' THEN 1 ELSE 0 END) AS obs,
              SUM(CASE WHEN type = 'action' THEN 1 ELSE 0 END) AS act,
              SUM(CASE WHEN type = 'outcome' THEN 1 ELSE 0 END) AS outc,
              SUM(CASE WHEN type = 'internal' THEN 1 ELSE 0 END) AS intr,
              SUM(CASE WHEN type = 'system' THEN 1 ELSE 0 END) AS sys
            FROM events
            """
        )
        r = cur.fetchone()
        return {
            "total": int(r["total"]),
            "observation": int(r["obs"]),
            "action": int(r["act"]),
            "outcome": int(r["outc"]),
            "internal": int(r["intr"]),
            "system": int(r["sys"]),
            "last_seq": self.last_seq(),
        }

    def verify_parent_links(self, sample_limit: int = 10000) -> Dict[str, Any]:
        sample_limit = max(1, int(sample_limit))
        cur = self._conn.execute(
            "SELECT id, parent_id FROM events WHERE parent_id IS NOT NULL LIMIT ?",
            (sample_limit,),
        )
        missing: List[Tuple[str, str]] = []

        for row in cur.fetchall():
            pid = row["parent_id"]
            exists = self._conn.execute(
                "SELECT 1 FROM events WHERE id = ? LIMIT 1",
                (pid,),
            ).fetchone()
            if not exists:
                missing.append((row["id"], pid))

        return {"checked": sample_limit, "missing_parent_links": missing}

    # ----------------------------
    # Forbidden Operations
    # ----------------------------

    def delete_all_events(self) -> None:
        raise MemoryIntegrityError(
            "Deletion of memory is forbidden by doctrine."
        )
