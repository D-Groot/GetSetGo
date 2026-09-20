# database/feedback_store.py
# Lightweight feedback-based learning layer.
#
# Every answer the user rates (👍 / 👎) nudges a persistent per-chunk
# "trust weight". Future retrievals add this weight on top of the raw
# cosine-similarity score, so sources the user has confirmed useful
# rank higher next time, and sources marked wrong/outdated/hallucinated
# sink down — a simple, transparent, explainable form of learning that
# doesn't require retraining any model.
#
# Storage: one SQLite file per user at ./db/{user_id}/feedback.sqlite3
# Two tables:
#   - doc_weights : running trust weight + up/down counters per chunk id
#   - feedback_log: full audit trail of every rating event (for diagnostics)

import sqlite3
import os
import time
from contextlib import contextmanager

from config import DB_BASE_PATH

UP_STEP      = 0.06   # weight nudge per 👍
DOWN_STEP    = 0.18   # weight nudge per 👎 (bigger — trust is easy to lose)
WEIGHT_MIN   = -0.5
WEIGHT_MAX   = 0.3

REASON_PENALTY = {
    "Wrong source pulled":   0.10,
    "Outdated information":  0.05,
    "Hallucination":         0.20,
}


def _db_path(user_id: str) -> str:
    folder = os.path.join(DB_BASE_PATH, user_id)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "feedback.sqlite3")


@contextmanager
def _conn(user_id: str):
    conn = sqlite3.connect(_db_path(user_id))
    conn.row_factory = sqlite3.Row
    try:
        _ensure_schema(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_schema(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS doc_weights (
            doc_id     TEXT PRIMARY KEY,
            weight     REAL DEFAULT 0.0,
            up_count   INTEGER DEFAULT 0,
            down_count INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  REAL,
            query      TEXT,
            rating     TEXT,
            reason     TEXT,
            doc_id     TEXT,
            source_name TEXT
        )
    """)


# ══════════════════════════════════════════════════════════════════════════════
#  WRITE — record a rating and update learned weights
# ══════════════════════════════════════════════════════════════════════════════

def record_feedback(user_id: str,
                     query: str,
                     sources: list,
                     rating: str,
                     reason: str = None) -> None:
    """
    rating  : "up" or "down"
    sources : list of source dicts used in the answer (must include 'doc_id',
              'source_name'); each gets its weight nudged.
    reason  : only for "down" — one of REASON_PENALTY keys, optional.
    """
    step = UP_STEP if rating == "up" else -(DOWN_STEP + REASON_PENALTY.get(reason, 0.0))

    with _conn(user_id) as conn:
        for src in sources:
            doc_id = src.get("doc_id")
            if not doc_id:
                continue

            row = conn.execute(
                "SELECT weight, up_count, down_count FROM doc_weights WHERE doc_id = ?",
                (doc_id,)
            ).fetchone()

            if row is None:
                new_weight = max(WEIGHT_MIN, min(WEIGHT_MAX, step))
                up, down = (1, 0) if rating == "up" else (0, 1)
                conn.execute(
                    "INSERT INTO doc_weights (doc_id, weight, up_count, down_count) VALUES (?, ?, ?, ?)",
                    (doc_id, new_weight, up, down)
                )
            else:
                new_weight = max(WEIGHT_MIN, min(WEIGHT_MAX, row["weight"] + step))
                up = row["up_count"] + (1 if rating == "up" else 0)
                down = row["down_count"] + (1 if rating == "down" else 0)
                conn.execute(
                    "UPDATE doc_weights SET weight = ?, up_count = ?, down_count = ? WHERE doc_id = ?",
                    (new_weight, up, down, doc_id)
                )

            conn.execute(
                "INSERT INTO feedback_log (timestamp, query, rating, reason, doc_id, source_name) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), query, rating, reason, doc_id, src.get("source_name"))
            )


# ══════════════════════════════════════════════════════════════════════════════
#  READ — apply learned weights at retrieval time
# ══════════════════════════════════════════════════════════════════════════════

def get_weights(user_id: str, doc_ids: list) -> dict:
    """Returns {doc_id: learned_weight} for the given ids (0.0 if never rated)."""
    if not doc_ids:
        return {}
    with _conn(user_id) as conn:
        placeholders = ",".join("?" * len(doc_ids))
        rows = conn.execute(
            f"SELECT doc_id, weight FROM doc_weights WHERE doc_id IN ({placeholders})",
            doc_ids
        ).fetchall()
    return {r["doc_id"]: r["weight"] for r in rows}


def get_stats(user_id: str) -> dict:
    """Aggregate stats for the Search Diagnostics drawer."""
    with _conn(user_id) as conn:
        totals = conn.execute(
            "SELECT COUNT(*) AS n, "
            "SUM(CASE WHEN rating = 'up' THEN 1 ELSE 0 END) AS ups, "
            "SUM(CASE WHEN rating = 'down' THEN 1 ELSE 0 END) AS downs "
            "FROM feedback_log"
        ).fetchone()
        top_boosted = conn.execute(
            "SELECT doc_id, weight, up_count, down_count FROM doc_weights "
            "ORDER BY weight DESC LIMIT 3"
        ).fetchall()
        top_penalized = conn.execute(
            "SELECT doc_id, weight, up_count, down_count FROM doc_weights "
            "ORDER BY weight ASC LIMIT 3"
        ).fetchall()
        reason_breakdown = conn.execute(
            "SELECT reason, COUNT(*) AS n FROM feedback_log "
            "WHERE rating = 'down' AND reason IS NOT NULL GROUP BY reason"
        ).fetchall()

    return {
        "total_ratings": totals["n"] or 0,
        "ups": totals["ups"] or 0,
        "downs": totals["downs"] or 0,
        "top_boosted": [dict(r) for r in top_boosted],
        "top_penalized": [dict(r) for r in top_penalized],
        "reason_breakdown": {r["reason"]: r["n"] for r in reason_breakdown},
    }
