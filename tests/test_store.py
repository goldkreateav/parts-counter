"""Тесты SQLite-хранилища счётчиков."""

from __future__ import annotations

from datetime import date

from conveyor_counter.counting.store import CountStore


def test_increment_and_get(tmp_path):
    store = CountStore(tmp_path / "counts.sqlite3")
    assert store.get_counts() == {}
    assert store.increment("floor_panel") == 1
    assert store.increment("floor_panel") == 2
    assert store.increment("other_part") == 1
    assert store.get_counts() == {"floor_panel": 2, "other_part": 1}
    store.close()


def test_counts_separated_by_day(tmp_path):
    store = CountStore(tmp_path / "counts.sqlite3")
    d1 = date(2026, 6, 10)
    d2 = date(2026, 6, 11)
    store.increment("floor_panel", d1)
    store.increment("floor_panel", d2)
    store.increment("floor_panel", d2)
    assert store.get_counts(d1) == {"floor_panel": 1}
    assert store.get_counts(d2) == {"floor_panel": 2}
    assert store.all_days() == [
        ("2026-06-10", "floor_panel", 1),
        ("2026-06-11", "floor_panel", 2),
    ]
    store.close()


def test_event_log(tmp_path):
    store = CountStore(tmp_path / "counts.sqlite3")
    store.log_event("spawned", 1, "floor_panel", 10)
    store.log_event("counted", 1, "floor_panel", 50)
    rows = store._conn.execute("SELECT kind, part_id, class_name, frame_idx FROM events").fetchall()
    assert rows == [("spawned", 1, "floor_panel", 10), ("counted", 1, "floor_panel", 50)]
    store.close()
