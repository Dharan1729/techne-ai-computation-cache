"""Persistent storage for learned actions (step 4 of the design).

An *action* is the reusable object: not an answer, but an operation. It holds a
safe expression that computes the result, the names of its parameters, and a
template for the reply. It is reached through one or more *phrasings*
(templates): the request with its values removed, such as
``What is {pct}% of {base}?``. Each phrasing is trusted separately, because
what was verified is "this wording means this operation", not just the
operation.

Phrasing status::

    probation  computes silently next to the LLM; agreements are counted
    active     answers requests (reached by agreement, or by approval)
    rejected   never answers; kept so the same mistake is not learned twice

Embeddings of the masked phrasings are stored with them, so a restart does not
re-embed anything. Storage is SQLite: one file, inspectable with any SQLite
tool and exportable to YAML for review.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

STATUSES = ("probation", "active", "rejected")
SCHEMA_VERSION = 2


@dataclass
class Template:
    text: str                         # the request with values replaced by {param}
    status: str = "probation"
    agree: int = 0                    # distinct parameter sets where the LLM agreed
    disagree: int = 0
    used: int = 0                     # times it answered a request
    note: str = ""
    id: int | None = None


@dataclass
class Action:
    name: str
    compute: str                      # safe expression over the parameter names
    params: list[str]
    templates: list[Template]
    response: str = "{value}"         # reply template: {value}, {param}
    enabled: bool = True              # false once every phrasing has been rejected
    note: str = ""
    id: int | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def template(self, text: str) -> Template | None:
        return next((t for t in self.templates if t.text == text), None)

    @property
    def status(self) -> str:
        """Summary for display: active if any phrasing answers requests."""
        if not self.enabled or all(t.status == "rejected" for t in self.templates):
            return "rejected"
        return "active" if any(t.status == "active" for t in self.templates) else "probation"

    @property
    def used(self) -> int:
        return sum(t.used for t in self.templates)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS actions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT UNIQUE NOT NULL,
  compute TEXT NOT NULL,
  response TEXT NOT NULL,
  params TEXT NOT NULL,
  enabled INTEGER NOT NULL DEFAULT 1,
  note TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS templates (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  action_id INTEGER NOT NULL REFERENCES actions(id) ON DELETE CASCADE,
  text TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'probation',
  disagree INTEGER NOT NULL DEFAULT 0,
  used INTEGER NOT NULL DEFAULT 0,
  note TEXT NOT NULL DEFAULT '',
  vec BLOB,
  UNIQUE(action_id, text)
);
CREATE TABLE IF NOT EXISTS observations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  template_id INTEGER NOT NULL REFERENCES templates(id) ON DELETE CASCADE,
  args TEXT NOT NULL,
  agrees INTEGER NOT NULL,
  ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS obs_template ON observations(template_id);
"""


class ActionStore:
    """Thread-safe SQLite store. ``path`` may be ``":memory:"``."""

    def __init__(self, path: str | Path = ":memory:", embedder_id: str = "") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.execute("PRAGMA foreign_keys = ON")
            if self.path != ":memory:":
                self._db.execute("PRAGMA journal_mode = WAL")
            self._db.executescript(_SCHEMA)
            row = self._db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
            if row is None:
                self._db.execute("INSERT INTO meta VALUES ('schema', ?)", (str(SCHEMA_VERSION),))
            elif int(row["value"]) != SCHEMA_VERSION:
                raise RuntimeError(f"store schema {row['value']} is not supported (expected {SCHEMA_VERSION})")
            # Embeddings are only valid for the model that made them. If the
            # embedder changed, drop the stored vectors; they are re-made lazily.
            old = self._db.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
            if embedder_id and (old is None or old["value"] != embedder_id):
                self._db.execute("UPDATE templates SET vec = NULL")
                self._db.execute("INSERT OR REPLACE INTO meta VALUES ('embedder', ?)", (embedder_id,))

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ---- reads ----------------------------------------------------------
    def _template(self, row: sqlite3.Row) -> Template:
        agree = self._db.execute(
            "SELECT COUNT(DISTINCT args) AS n FROM observations WHERE template_id=? AND agrees=1", (row["id"],)
        ).fetchone()["n"]
        return Template(row["text"], row["status"], agree, row["disagree"], row["used"], row["note"], row["id"])

    def _action(self, row: sqlite3.Row) -> Action:
        tmpl = [self._template(r) for r in self._db.execute(
            "SELECT * FROM templates WHERE action_id=? ORDER BY id", (row["id"],))]
        return Action(name=row["name"], compute=row["compute"], params=json.loads(row["params"]), templates=tmpl,
                      response=row["response"], enabled=bool(row["enabled"]), note=row["note"], id=row["id"],
                      created_at=row["created_at"], updated_at=row["updated_at"])

    def get(self, name: str) -> Action | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM actions WHERE name=?", (name,)).fetchone()
            return self._action(row) if row else None

    def list(self) -> list[Action]:
        with self._lock:
            return [self._action(r) for r in self._db.execute("SELECT * FROM actions ORDER BY id")]

    def vectors(self, name: str) -> list[np.ndarray | None]:
        """Stored embedding for each phrasing of ``name`` (in order), None where missing."""
        with self._lock:
            rows = self._db.execute(
                "SELECT t.vec FROM templates t JOIN actions a ON a.id=t.action_id WHERE a.name=? ORDER BY t.id",
                (name,)).fetchall()
        return [np.frombuffer(r["vec"], dtype=np.float32).copy() if r["vec"] is not None else None for r in rows]

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) AS n FROM actions").fetchone()["n"]

    # ---- writes ---------------------------------------------------------
    def add(self, action: Action, vecs: list[np.ndarray] | None = None) -> Action:
        """Insert a new action. Raises ``ValueError`` if the name exists."""
        with self._lock:
            if self._db.execute("SELECT 1 FROM actions WHERE name=?", (action.name,)).fetchone():
                raise ValueError(f"action {action.name!r} already exists")
            now = time.time()
            aid = self._db.execute(
                "INSERT INTO actions(name, compute, response, params, enabled, note, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (action.name, action.compute, action.response, json.dumps(action.params), int(action.enabled),
                 action.note, now, now)).lastrowid
            for i, t in enumerate(action.templates):
                v = vecs[i].astype(np.float32).tobytes() if vecs is not None else None
                self._db.execute("INSERT OR IGNORE INTO templates(action_id, text, status, note, vec) VALUES (?,?,?,?,?)",
                                 (aid, t.text, t.status, t.note, v))
            return self.get(action.name)  # type: ignore[return-value]

    def add_template(self, name: str, text: str, vec: np.ndarray | None, status: str = "probation",
                     note: str = "") -> bool:
        """Add another phrasing to an action. Returns False if it was already there."""
        with self._lock:
            row = self._db.execute("SELECT id FROM actions WHERE name=?", (name,)).fetchone()
            if row is None:
                raise KeyError(name)
            n = self._db.execute(
                "INSERT OR IGNORE INTO templates(action_id, text, status, note, vec) VALUES (?,?,?,?,?)",
                (row["id"], text, status, note, vec.astype(np.float32).tobytes() if vec is not None else None)).rowcount
            self._db.execute("UPDATE actions SET updated_at=? WHERE id=?", (time.time(), row["id"]))
            return bool(n)

    def set_vectors(self, name: str, texts: list[str], vecs: np.ndarray) -> None:
        with self._lock:
            row = self._db.execute("SELECT id FROM actions WHERE name=?", (name,)).fetchone()
            for t, v in zip(texts, vecs):
                self._db.execute("UPDATE templates SET vec=? WHERE action_id=? AND text=?",
                                 (v.astype(np.float32).tobytes(), row["id"], t))

    def _tid(self, name: str, text: str) -> int:
        row = self._db.execute(
            "SELECT t.id FROM templates t JOIN actions a ON a.id=t.action_id WHERE a.name=? AND t.text=?",
            (name, text)).fetchone()
        if row is None:
            raise KeyError((name, text))
        return row["id"]

    def set_template_status(self, name: str, text: str, status: str, note: str | None = None) -> None:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        with self._lock:
            tid = self._tid(name, text)
            self._db.execute("UPDATE templates SET status=?, note=COALESCE(?, note) WHERE id=?", (status, note, tid))
            alive = self._db.execute(
                "SELECT COUNT(*) AS n FROM templates t JOIN actions a ON a.id=t.action_id "
                "WHERE a.name=? AND t.status!='rejected'", (name,)).fetchone()["n"]
            self._db.execute("UPDATE actions SET enabled=?, updated_at=? WHERE name=?", (int(alive > 0), time.time(), name))

    def set_action_status(self, name: str, status: str, note: str | None = None) -> None:
        """Set every phrasing of an action at once (review commands)."""
        a = self.get(name)
        if a is None:
            raise KeyError(name)
        for t in a.templates:
            if t.status != "rejected" or status != "active":
                self.set_template_status(name, t.text, status, note)

    def record_used(self, name: str, text: str) -> None:
        with self._lock:
            self._db.execute("UPDATE templates SET used=used+1 WHERE id=?", (self._tid(name, text),))

    def record_observation(self, name: str, text: str, args: dict, agrees: bool) -> tuple[int, int]:
        """Log one comparison with the LLM for a phrasing. Returns (distinct
        agreeing parameter sets, disagreements) so far."""
        with self._lock:
            tid = self._tid(name, text)
            self._db.execute("INSERT INTO observations(template_id, args, agrees, ts) VALUES (?,?,?,?)",
                             (tid, json.dumps(args, sort_keys=True), int(agrees), time.time()))
            if not agrees:
                self._db.execute("UPDATE templates SET disagree=disagree+1 WHERE id=?", (tid,))
            agree = self._db.execute(
                "SELECT COUNT(DISTINCT args) AS n FROM observations WHERE template_id=? AND agrees=1", (tid,)
            ).fetchone()["n"]
            dis = self._db.execute("SELECT disagree FROM templates WHERE id=?", (tid,)).fetchone()["disagree"]
            return agree, dis

    def delete(self, name: str) -> bool:
        with self._lock:
            return bool(self._db.execute("DELETE FROM actions WHERE name=?", (name,)).rowcount)

    # ---- review and portability ------------------------------------------
    def export(self) -> list[dict[str, Any]]:
        out = []
        for a in self.list():
            out.append({"name": a.name, "compute": a.compute, "params": a.params, "response": a.response,
                        "note": a.note, "enabled": a.enabled,
                        "templates": [{"text": t.text, "status": t.status, "note": t.note} for t in a.templates]})
        return out

    def import_actions(self, items: list[dict[str, Any]], overwrite: bool = False) -> tuple[int, int]:
        """Add actions from ``export()`` output. Returns (added, skipped). Evidence
        counters are not imported: an imported phrasing keeps its status as given."""
        added = skipped = 0
        for d in items:
            if self.get(d["name"]):
                if not overwrite:
                    skipped += 1
                    continue
                self.delete(d["name"])
            self.add(Action(
                name=d["name"], compute=d["compute"], params=list(d["params"]), response=d.get("response", "{value}"),
                note=d.get("note", ""), enabled=d.get("enabled", True),
                templates=[Template(t["text"], t.get("status", "probation"), note=t.get("note", ""))
                           for t in d["templates"]]))
            added += 1
        return added, skipped
