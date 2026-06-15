"""
SQLite persistence layer for the Icepak manifold agent.
Single file = single source of truth. No in-memory state.
"""

import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "icepak_manifold.db"

# ── Connection ─────────────────────────────────────────────────────────
def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    return c

# ── Schema ─────────────────────────────────────────────────────────────
def init_db():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS run_metadata (
            id                     INTEGER PRIMARY KEY CHECK (id = 1),
            batch_size             INTEGER NOT NULL DEFAULT 16,
            alpha                  REAL    NOT NULL DEFAULT 0.85,
            current_explore_ratio  REAL    NOT NULL DEFAULT 0.30,
            previous_explore_ratio REAL    NOT NULL DEFAULT 0.30,
            next_batch_number      INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS dof_configurations (
            param_name  TEXT    PRIMARY KEY,
            min_val     REAL    NOT NULL,
            max_val     REAL    NOT NULL,
            step_size   REAL    NOT NULL,
            param_order INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS batches (
            batch_number  INTEGER PRIMARY KEY,
            source        TEXT    NOT NULL,
            created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
            explore_ratio REAL,
            n_points      INTEGER NOT NULL DEFAULT 0,
            status        TEXT    NOT NULL DEFAULT 'staged'
        );

        CREATE TABLE IF NOT EXISTS points (
            point_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_number      INTEGER NOT NULL REFERENCES batches(batch_number),
            dof_N             REAL,
            dof_w             REAL,
            dof_tb            REAL,
            dof_H             REAL,
            status            TEXT    NOT NULL DEFAULT 'STAGED',
            performance_score REAL,
            disabled          INTEGER NOT NULL DEFAULT 0
        );
        """)

        # Migrations for older DBs
        for _sql in [
            "ALTER TABLE points ADD COLUMN disabled      INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE points ADD COLUMN manifold_pred REAL",
        ]:
            try:
                c.execute(_sql)
            except Exception:
                pass  # column already exists

def reset_db():
    with _conn() as c:
        c.executescript("""
        DELETE FROM points;
        DELETE FROM batches;
        DELETE FROM run_metadata;
        DELETE FROM dof_configurations;
        """)

# ── run_metadata ───────────────────────────────────────────────────────
def get_metadata():
    with _conn() as c:
        row = c.execute("SELECT * FROM run_metadata WHERE id=1").fetchone()
        return dict(row) if row else None

def upsert_metadata(**kwargs):
    with _conn() as c:
        if c.execute("SELECT 1 FROM run_metadata WHERE id=1").fetchone():
            sets = ", ".join(f"{k}=?" for k in kwargs)
            c.execute(f"UPDATE run_metadata SET {sets} WHERE id=1",
                      list(kwargs.values()))
        else:
            kwargs['id'] = 1
            cols = ", ".join(kwargs.keys())
            vals = ", ".join("?" * len(kwargs))
            c.execute(f"INSERT INTO run_metadata ({cols}) VALUES ({vals})",
                      list(kwargs.values()))

# ── dof_configurations ─────────────────────────────────────────────────
def set_dof_configs(configs):
    with _conn() as c:
        c.execute("DELETE FROM dof_configurations")
        for i, cfg in enumerate(configs):
            c.execute(
                "INSERT INTO dof_configurations VALUES (?,?,?,?,?)",
                (cfg['param_name'], cfg['min_val'], cfg['max_val'],
                 cfg['step_size'], i)
            )

def get_dof_configs():
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM dof_configurations ORDER BY param_order"
        ).fetchall()
        return [dict(r) for r in rows]

# ── batches ────────────────────────────────────────────────────────────
def get_batches():
    with _conn() as c:
        rows = c.execute("SELECT * FROM batches ORDER BY batch_number").fetchall()
        return [dict(r) for r in rows]

def _ensure_batch(batch_number, source, status, explore_ratio, n_points):
    """Insert batch row if it doesn't exist; update n_points and status if it does."""
    with _conn() as c:
        if c.execute("SELECT 1 FROM batches WHERE batch_number=?", (batch_number,)).fetchone():
            c.execute(
                "UPDATE batches SET status=?, n_points=? WHERE batch_number=?",
                (status, n_points, batch_number)
            )
        else:
            c.execute(
                """INSERT INTO batches (batch_number, source, created_at, explore_ratio, n_points, status)
                   VALUES (?,?,?,?,?,?)""",
                (batch_number, source,
                 datetime.now().isoformat(timespec='seconds'),
                 explore_ratio, n_points, status)
            )

# ── points — agent workflow ────────────────────────────────────────────
def insert_staged(batch_number, points, explore_ratio=None, source='agent'):
    """Stage a batch. points: (n,4) array [N, w, tb, H]."""
    n = len(points)
    _ensure_batch(batch_number, source, 'staged', explore_ratio, n)
    with _conn() as c:
        c.executemany(
            """INSERT INTO points (batch_number, dof_N, dof_w, dof_tb, dof_H, status)
               VALUES (?,?,?,?,?,'STAGED')""",
            [(batch_number, float(r[0]), float(r[1]),
              float(r[2]), float(r[3])) for r in points]
        )

def discard_staged():
    with _conn() as c:
        c.execute("DELETE FROM points WHERE status='STAGED'")
        c.execute("DELETE FROM batches WHERE status='staged'")

def commit_scores(point_id_score_pairs):
    """Flip STAGED points to EVALUATED and update the owning batch status."""
    with _conn() as c:
        c.executemany(
            """UPDATE points SET status='EVALUATED', performance_score=?
               WHERE point_id=?""",
            [(score, pid) for pid, score in point_id_score_pairs]
        )
        # Flip batches that no longer have any STAGED points
        c.execute("""
            UPDATE batches SET status='evaluated'
            WHERE status='staged'
              AND batch_number NOT IN (
                  SELECT DISTINCT batch_number FROM points WHERE status='STAGED'
              )
        """)

# ── points — seeding ───────────────────────────────────────────────────
def seed_batch(batch_number, source, points_with_scores):
    """
    Insert a fully-evaluated batch for historical seeding.
    points_with_scores: list/array of (N, w, tb, H, T_chip)
    source: 'seed_sweep' | 'seed_sparse' | 'seed_agent'
    """
    n = len(points_with_scores)
    _ensure_batch(batch_number, source, 'evaluated', None, n)
    with _conn() as c:
        c.executemany(
            """INSERT INTO points
               (batch_number, dof_N, dof_w, dof_tb, dof_H, status, performance_score)
               VALUES (?,?,?,?,?,'EVALUATED',?)""",
            [(batch_number, float(r[0]), float(r[1]),
              float(r[2]), float(r[3]), float(r[4])) for r in points_with_scores]
        )

# ── queries ────────────────────────────────────────────────────────────
def get_evaluated():
    """Returns only enabled (disabled=0) evaluated points — used by all math functions."""
    with _conn() as c:
        rows = c.execute(
            """SELECT point_id, batch_number, dof_N, dof_w, dof_tb, dof_H, performance_score
               FROM points WHERE status='EVALUATED' AND disabled=0
               ORDER BY point_id"""
        ).fetchall()
        return [dict(r) for r in rows]

def get_all_evaluated():
    """Returns all evaluated points including disabled ones — used by the history UI."""
    with _conn() as c:
        rows = c.execute(
            """SELECT point_id, batch_number, dof_N, dof_w, dof_tb, dof_H,
                      performance_score, disabled, manifold_pred
               FROM points WHERE status='EVALUATED'
               ORDER BY point_id"""
        ).fetchall()
        return [dict(r) for r in rows]

def set_points_disabled(updates):
    """updates: list of (point_id, disabled) — 0=enabled, 1=disabled."""
    with _conn() as c:
        c.executemany(
            "UPDATE points SET disabled=? WHERE point_id=?",
            [(int(d), pid) for pid, d in updates]
        )

def set_manifold_preds(updates):
    """updates: list of (point_id, pred_value)."""
    with _conn() as c:
        c.executemany(
            "UPDATE points SET manifold_pred=? WHERE point_id=?",
            [(float(v), pid) for pid, v in updates]
        )

def get_staged():
    with _conn() as c:
        rows = c.execute(
            """SELECT point_id, batch_number, dof_N, dof_w, dof_tb, dof_H
               FROM points WHERE status='STAGED'
               ORDER BY point_id"""
        ).fetchall()
        return [dict(r) for r in rows]

def stats():
    with _conn() as c:
        n_eval   = c.execute("SELECT COUNT(*) FROM points WHERE status='EVALUATED'").fetchone()[0]
        n_staged = c.execute("SELECT COUNT(*) FROM points WHERE status='STAGED'").fetchone()[0]
        best_T   = c.execute(
            "SELECT MIN(performance_score) FROM points WHERE status='EVALUATED'"
        ).fetchone()[0]
        return {'evaluated': n_eval, 'staged': n_staged, 'best_T': best_T}
