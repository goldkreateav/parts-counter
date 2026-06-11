"""Хранилище счётчиков деталей по дням (SQLite, стандартная библиотека)."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path


class CountStore:
    """Счётчики `день x класс детали` плюс журнал событий конвейера."""

    def __init__(self, db_path: str | Path):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_counts (
                day TEXT NOT NULL,
                class_name TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (day, class_name)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                ts TEXT NOT NULL,
                kind TEXT NOT NULL,
                part_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                frame_idx INTEGER NOT NULL
            )
            """
        )
        self._conn.commit()

    def increment(self, class_name: str, day: date | None = None) -> int:
        """Увеличивает счётчик класса за день, возвращает новое значение."""
        day_str = (day or date.today()).isoformat()
        self._conn.execute(
            """
            INSERT INTO daily_counts (day, class_name, count) VALUES (?, ?, 1)
            ON CONFLICT (day, class_name) DO UPDATE SET count = count + 1
            """,
            (day_str, class_name),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT count FROM daily_counts WHERE day = ? AND class_name = ?",
            (day_str, class_name),
        ).fetchone()
        return int(row[0])

    def get_counts(self, day: date | None = None) -> dict[str, int]:
        day_str = (day or date.today()).isoformat()
        rows = self._conn.execute(
            "SELECT class_name, count FROM daily_counts WHERE day = ?", (day_str,)
        ).fetchall()
        return {name: int(count) for name, count in rows}

    def all_days(self) -> list[tuple[str, str, int]]:
        return [
            (day, name, int(count))
            for day, name, count in self._conn.execute(
                "SELECT day, class_name, count FROM daily_counts ORDER BY day, class_name"
            ).fetchall()
        ]

    def log_event(self, kind: str, part_id: int, class_name: str, frame_idx: int) -> None:
        self._conn.execute(
            "INSERT INTO events (ts, kind, part_id, class_name, frame_idx) VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), kind, part_id, class_name, frame_idx),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
