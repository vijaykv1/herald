"""Per-user daily message quota tracking."""

import sqlite3
from datetime import datetime, timezone
from fastapi import HTTPException

USAGE_DB_PATH = "herald_usage.db"
DAILY_MESSAGE_LIMIT = 15


class UsageTracker:
    """Tracks per-user daily message usage in a persistent SQLite database."""

    def __init__(self, db_path: str = USAGE_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the usage table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_usage (
                    user_id TEXT NOT NULL,
                    date    TEXT NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (user_id, date)
                )
            """)

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def get_count(self, user_id: str) -> int:
        """Return how many messages the user has sent today."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT message_count FROM user_usage WHERE user_id = ? AND date = ?",
                (user_id, self._today()),
            ).fetchone()
        return row[0] if row else 0

    def increment(self, user_id: str) -> int:
        """Increment today's count and return the new value."""
        today = self._today()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO user_usage (user_id, date, message_count) VALUES (?, ?, 1)
                ON CONFLICT(user_id, date) DO UPDATE SET message_count = message_count + 1
                """,
                (user_id, today),
            )
            row = conn.execute(
                "SELECT message_count FROM user_usage WHERE user_id = ? AND date = ?",
                (user_id, today),
            ).fetchone()
        return row[0]

    def check_quota(self, user_id: str) -> tuple[int, int]:
        """Return (used, remaining). Raise HTTP 429 if the daily limit is reached."""
        used = self.get_count(user_id)
        remaining = max(0, DAILY_MESSAGE_LIMIT - used)
        if used >= DAILY_MESSAGE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "daily_limit_reached",
                    "message": (
                        f"You've reached your daily limit of {DAILY_MESSAGE_LIMIT} messages. "
                        "Come back tomorrow!"
                    ),
                    "limit": DAILY_MESSAGE_LIMIT,
                    "used": used,
                    "remaining": 0,
                },
            )
        return used, remaining
