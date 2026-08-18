# database.py - Session storage
# Location: project_root/database.py
#
# Sessions are kept in SQLite, scoped to an anonymous device ID held in a
# cookie. No accounts, no personal data, and nothing here that a free
# PythonAnywhere worker cannot run.
#
# The file lives next to this module so the path is stable no matter what
# working directory the WSGI server starts in.

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

DB_PATH = os.environ.get(
    'SHAPEFORM_DB',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sessions.db')
)

# Keep a lid on abuse: one device cannot store an unbounded number of rows.
MAX_SESSIONS_PER_DEVICE = 500


@contextmanager
def get_db():
    """A connection with rows that behave like dicts."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create the schema. Safe to call on every boot."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id           TEXT PRIMARY KEY,
                device_id    TEXT NOT NULL,
                exercise     TEXT NOT NULL,
                mode         TEXT NOT NULL DEFAULT 'webcam',
                started_at   TEXT NOT NULL,
                duration_sec INTEGER NOT NULL DEFAULT 0,
                reps         INTEGER NOT NULL DEFAULT 0,
                good_reps    INTEGER NOT NULL DEFAULT 0,
                bad_reps     INTEGER NOT NULL DEFAULT 0,
                faults       TEXT NOT NULL DEFAULT '{}',
                created_at   TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_device
            ON sessions (device_id, started_at DESC)
        """)


def new_device_id():
    return uuid.uuid4().hex


def _row_to_session(row):
    reps = row['reps'] or 0
    good = row['good_reps'] or 0
    try:
        faults = json.loads(row['faults'])
    except (ValueError, TypeError):
        faults = {}

    return {
        'id': row['id'],
        'exercise': row['exercise'],
        'mode': row['mode'],
        'startedAt': row['started_at'],
        'durationSec': row['duration_sec'],
        'reps': reps,
        'good': good,
        'bad': row['bad_reps'] or 0,
        'accuracy': round((good / reps) * 100) if reps else 0,
        'faults': faults,
    }


def create_session(device_id, payload):
    """Insert one finished set. Returns the stored session."""
    reps = max(0, int(payload.get('reps') or 0))
    good = max(0, min(reps, int(payload.get('good') or 0)))
    bad = max(0, min(reps, int(payload.get('bad') or (reps - good))))

    faults = payload.get('faults') or {}
    if not isinstance(faults, dict):
        faults = {}
    # Coerce to {str: int} so a malformed client cannot poison the row.
    faults = {str(k): int(v) for k, v in faults.items()
              if isinstance(v, (int, float)) and v > 0}

    started = payload.get('startedAt') or datetime.now(timezone.utc).isoformat()

    session_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        conn.execute("""
            INSERT INTO sessions
                (id, device_id, exercise, mode, started_at, duration_sec,
                 reps, good_reps, bad_reps, faults, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, device_id,
            str(payload.get('exercise') or 'unknown'),
            str(payload.get('mode') or 'webcam'),
            str(started),
            max(0, int(payload.get('durationSec') or 0)),
            reps, good, bad, json.dumps(faults), now,
        ))

        # Trim the oldest rows past the cap.
        conn.execute("""
            DELETE FROM sessions
            WHERE device_id = ?
              AND id NOT IN (
                  SELECT id FROM sessions
                  WHERE device_id = ?
                  ORDER BY started_at DESC
                  LIMIT ?
              )
        """, (device_id, device_id, MAX_SESSIONS_PER_DEVICE))

    return get_session(device_id, session_id)


def get_session(device_id, session_id):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE device_id = ? AND id = ?",
            (device_id, session_id)
        ).fetchone()
    return _row_to_session(row) if row else None


def list_sessions(device_id, exercise=None, limit=200):
    """Newest first, optionally filtered to one exercise."""
    sql = "SELECT * FROM sessions WHERE device_id = ?"
    args = [device_id]
    if exercise and exercise != 'all':
        sql += " AND exercise = ?"
        args.append(exercise)
    sql += " ORDER BY started_at DESC LIMIT ?"
    args.append(int(limit))

    with get_db() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [_row_to_session(r) for r in rows]


def delete_session(device_id, session_id):
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM sessions WHERE device_id = ? AND id = ?",
            (device_id, session_id)
        )
    return cur.rowcount > 0


def clear_sessions(device_id):
    with get_db() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE device_id = ?", (device_id,))
    return cur.rowcount


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _parse_dt(value):
    """Parse an ISO timestamp into an aware datetime, tolerating a
    trailing Z and naive values written by older clients."""
    try:
        text = str(value).replace('Z', '+00:00')
        dt = datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def compute_stats(device_id):
    """Everything the dashboard needs, in one pass over the rows."""
    sessions = list_sessions(device_id, limit=MAX_SESSIONS_PER_DEVICE)

    totals = {
        'sessions': len(sessions),
        'reps': 0,
        'good': 0,
        'bad': 0,
        'seconds': 0,
        'accuracy': 0,
        'streak': 0,
        'faults': {},
        'byExercise': {},
        'personalBests': {},
    }

    days = set()
    for s in sessions:
        totals['reps'] += s['reps']
        totals['good'] += s['good']
        totals['bad'] += s['bad']
        totals['seconds'] += s['durationSec']

        for key, count in s['faults'].items():
            totals['faults'][key] = totals['faults'].get(key, 0) + count

        ex = totals['byExercise'].setdefault(
            s['exercise'], {'reps': 0, 'good': 0, 'sessions': 0, 'best': 0}
        )
        ex['reps'] += s['reps']
        ex['good'] += s['good']
        ex['sessions'] += 1
        ex['best'] = max(ex['best'], s['reps'])

        best = totals['personalBests'].get(s['exercise'], 0)
        totals['personalBests'][s['exercise']] = max(best, s['reps'])

        dt = _parse_dt(s['startedAt'])
        if dt:
            days.add(dt.astimezone().date())

    totals['accuracy'] = round((totals['good'] / totals['reps']) * 100) if totals['reps'] else 0
    totals['streak'] = _streak_from_days(days)
    return totals


def _streak_from_days(days):
    """Consecutive training days, counting back from today. A streak
    survives if the last session was yesterday."""
    if not days:
        return 0

    today = datetime.now().astimezone().date()
    cursor = today
    if cursor not in days:
        cursor = today - timedelta(days=1)
        if cursor not in days:
            return 0

    streak = 0
    while cursor in days:
        streak += 1
        cursor -= timedelta(days=1)
    return streak
