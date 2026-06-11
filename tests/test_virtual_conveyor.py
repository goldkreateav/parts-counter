"""Тесты логики виртуального конвейера (без зависимостей от torch/cv2)."""

from __future__ import annotations

from conveyor_counter.config import ConveyorConfig
from conveyor_counter.counting.virtual_conveyor import Detection, VirtualConveyor

CFG = ConveyorConfig(
    spawn_zone_x=0.7,
    count_line_x=0.2,
    match_gate=0.18,
    right_tolerance=0.04,
    min_spawn_separation=0.25,
    max_missed_frames=40,
    max_coast_frames=12,
    velocity_smoothing=0.3,
    default_velocity=-0.02,
)

SPEED = -0.02  # реальная скорость детали: 2% ширины кадра за кадр


def det(x: float, cls: str = "floor_panel") -> Detection:
    return Detection(x=x, y=0.5, w=0.2, h=0.4, class_name=cls, confidence=0.9)


def run_frames(conveyor: VirtualConveyor, frames: list[list[Detection]]):
    events = []
    for idx, dets in enumerate(frames):
        events.extend(conveyor.update(dets, idx))
    return events


def kinds(events, kind):
    return [e for e in events if e.kind == kind]


def test_single_part_counted_once():
    """Деталь проезжает справа налево и засчитывается ровно один раз."""
    conveyor = VirtualConveyor(CFG)
    frames = [[det(0.9 + SPEED * i)] for i in range(40)]
    events = run_frames(conveyor, frames)
    assert len(kinds(events, "spawned")) == 1
    assert len(kinds(events, "counted")) == 1
    assert len(kinds(events, "removed")) == 0
    assert conveyor.parts == []


def test_yolo_flicker_does_not_duplicate():
    """Пролаги YOLO: периодические детекции на старой позиции справа не плодят
    новые экземпляры и не двигают деталь вправо."""
    conveyor = VirtualConveyor(CFG)
    frames = []
    for i in range(40):
        x = 0.9 + SPEED * i
        dets = [det(x)]
        if i % 5 == 0 and i > 0:
            dets.append(det(min(0.95, x + 0.12)))  # «фантом» правее текущей позиции
        frames.append(dets)
    events = run_frames(conveyor, frames)
    assert len(kinds(events, "spawned")) == 1
    assert len(kinds(events, "counted")) == 1


def test_position_never_moves_right():
    """Детекция чуть правее (в пределах допуска) не сдвигает экземпляр вправо."""
    conveyor = VirtualConveyor(CFG)
    conveyor.update([det(0.9)], 0)
    part = conveyor.parts[0]
    conveyor.update([det(0.93)], 1)  # правее на 0.03 < right_tolerance — матчится, но x не растёт
    assert part.x <= 0.9


def test_detection_gap_coasting_still_counts():
    """Пропуск детекций в середине пути: экземпляр докатывается и засчитывается один раз."""
    conveyor = VirtualConveyor(CFG)
    frames: list[list[Detection]] = []
    for i in range(45):
        x = 0.9 + SPEED * i
        if 12 <= i < 20:
            frames.append([])  # YOLO молчит 8 кадров
        else:
            frames.append([det(x)])
    events = run_frames(conveyor, frames)
    assert len(kinds(events, "spawned")) == 1
    assert len(kinds(events, "counted")) == 1


def test_crossing_during_short_gap_counts():
    """Деталь пересекает линию «накатом» сразу после пропажи детекций — засчитываем."""
    conveyor = VirtualConveyor(CFG)
    frames: list[list[Detection]] = []
    for i in range(50):
        x = 0.9 + SPEED * i
        frames.append([det(x)] if x > 0.3 else [])  # YOLO слепнет у самой линии
    events = run_frames(conveyor, frames)
    assert len(kinds(events, "counted")) == 1


def test_removed_part_not_counted():
    """Деталь убрали с ленты задолго до линии (брак) — счётчик не растёт."""
    conveyor = VirtualConveyor(CFG)
    frames: list[list[Detection]] = []
    for i in range(70):
        x = 0.9 + SPEED * i
        frames.append([det(x)] if x > 0.55 else [])  # дальше деталь физически исчезла
    events = run_frames(conveyor, frames)
    assert len(kinds(events, "counted")) == 0
    assert len(kinds(events, "removed")) == 1
    assert conveyor.parts == []


def test_two_parts_sequential():
    """Две детали подряд — два спавна и два подсчёта."""
    conveyor = VirtualConveyor(CFG)
    frames: list[list[Detection]] = []
    for i in range(70):
        dets = []
        x1 = 0.9 + SPEED * i
        if x1 > 0.1:
            dets.append(det(x1))
        x2 = 0.9 + SPEED * (i - 25)  # вторая деталь въезжает на 25 кадров позже
        if i >= 25 and x2 > 0.1:
            dets.append(det(x2))
        frames.append(dets)
    events = run_frames(conveyor, frames)
    assert len(kinds(events, "spawned")) == 2
    assert len(kinds(events, "counted")) == 2


def test_detection_left_of_line_after_count_ignored():
    """После подсчёта детекции левее линии не создают экземпляров и не считаются."""
    conveyor = VirtualConveyor(CFG)
    frames = [[det(0.9 + SPEED * i)] for i in range(40)]
    events = run_frames(conveyor, frames)
    assert len(kinds(events, "counted")) == 1
    # «Хвост» детекций уже за линией.
    more = []
    for i in range(40, 50):
        more.extend(conveyor.update([det(0.1)], i))
    assert more == []
    assert conveyor.parts == []


def test_mid_belt_ghost_ignored():
    """Одиночная ложная детекция в середине ленты (вне зоны спавна) игнорируется."""
    conveyor = VirtualConveyor(CFG)
    events = conveyor.update([det(0.5)], 0)
    assert events == []
    assert conveyor.parts == []


def test_different_classes_tracked_separately():
    """Детали разных классов не матчятся друг с другом."""
    conveyor = VirtualConveyor(CFG)
    conveyor.update([det(0.9, "panel_a")], 0)
    events = conveyor.update([det(0.88, "panel_b")], 1)
    # panel_b не сматчилась с panel_a; она в зоне спавна, но рядом уже есть
    # экземпляр другого класса — допускаем спавн, т.к. разделение по классам.
    assert len(conveyor.parts) == 2
    assert {p.class_name for p in conveyor.parts} == {"panel_a", "panel_b"}
    assert len(kinds(events, "spawned")) == 1
