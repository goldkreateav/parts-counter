"""Виртуальная модель конвейера.

Конвейер движется справа налево. Детекции YOLO не считаются напрямую —
они лишь обновляют состояние виртуальных экземпляров деталей:

* экземпляр создаётся только когда детекция появляется в зоне спавна справа;
* экземпляр может двигаться ТОЛЬКО влево — детекции правее его текущей позиции
  (за пределами небольшого допуска) считаются пролагами YOLO и игнорируются;
* при пропусках детекций экземпляр «докатывается» по своей оценённой скорости;
* при пересечении линии подсчёта экземпляр засчитывается и исчезает с ленты,
  дальнейшие детекции левее линии ни с чем не сопоставляются и не плодят счёт;
* если деталь пропала надолго до линии (убрали с ленты как брак) — экземпляр
  удаляется без подсчёта.

Все X-координаты нормированы в [0, 1] относительно ширины кадра.
Модуль не зависит от ultralytics/cv2 — его можно тестировать в чистом Python.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from conveyor_counter.config import ConveyorConfig

# Скорость не может быть положительной (вправо) и не может быть быстрее
# четверти кадра за тик — защита от выбросов при ошибочном сопоставлении.
_MIN_VELOCITY = -0.25
_MAX_VELOCITY = 0.0


@dataclass
class Detection:
    """Одна детекция YOLO в кадре (координаты нормированы)."""

    x: float  # центр бокса по X
    y: float  # центр бокса по Y
    w: float = 0.0
    h: float = 0.0
    class_name: str = ""
    confidence: float = 1.0


@dataclass
class PartInstance:
    """Виртуальный экземпляр детали на ленте."""

    part_id: int
    class_name: str
    x: float
    y: float
    velocity: float  # нормированная ширина кадра за кадр, <= 0
    created_frame: int
    last_seen_frame: int
    hits: int = 1
    last_box: tuple[float, float] = (0.0, 0.0)  # (w, h) последней детекции

    def missed_for(self, frame_idx: int) -> int:
        return frame_idx - self.last_seen_frame


@dataclass
class ConveyorEvent:
    """Событие виртуального конвейера."""

    kind: str  # "spawned" | "counted" | "removed"
    part_id: int
    class_name: str
    frame_idx: int
    x: float


@dataclass
class VirtualConveyor:
    config: ConveyorConfig = field(default_factory=ConveyorConfig)
    parts: list[PartInstance] = field(default_factory=list)
    _id_counter: itertools.count = field(default_factory=lambda: itertools.count(1))
    _belt_velocity: float | None = None  # EMA скорости ленты по всем деталям

    @property
    def belt_velocity(self) -> float:
        if self._belt_velocity is None:
            return self.config.default_velocity
        return self._belt_velocity

    def update(self, detections: list[Detection], frame_idx: int) -> list[ConveyorEvent]:
        """Обрабатывает детекции очередного кадра, возвращает события."""
        cfg = self.config
        events: list[ConveyorEvent] = []

        matched_parts = self._match(detections, frame_idx)

        # Несопоставленные экземпляры катятся влево по своей скорости.
        for part in self.parts:
            if part not in matched_parts:
                part.x += part.velocity

        # Новые детали появляются только в зоне спавна справа.
        for det in detections:
            if id(det) in self._matched_detections:
                continue
            if det.x < cfg.spawn_zone_x:
                continue  # пролаг/призрак в середине ленты — игнорируем
            too_close = any(
                abs(det.x - p.x) < cfg.min_spawn_separation and p.class_name == det.class_name
                for p in self.parts
            )
            if too_close:
                continue
            part = PartInstance(
                part_id=next(self._id_counter),
                class_name=det.class_name,
                x=det.x,
                y=det.y,
                velocity=self.belt_velocity,
                created_frame=frame_idx,
                last_seen_frame=frame_idx,
                last_box=(det.w, det.h),
            )
            self.parts.append(part)
            events.append(ConveyorEvent("spawned", part.part_id, part.class_name, frame_idx, part.x))

        # Подсчёт пересечений линии и удаление пропавших деталей.
        survivors: list[PartInstance] = []
        for part in self.parts:
            missed = part.missed_for(frame_idx)
            if part.x <= cfg.count_line_x:
                # Засчитываем, только если деталь видели недавно: иначе она
                # доехала до линии чистым «накатом» — скорее всего её убрали.
                kind = "counted" if missed <= cfg.max_coast_frames else "removed"
                events.append(ConveyorEvent(kind, part.part_id, part.class_name, frame_idx, part.x))
            elif missed > cfg.max_missed_frames:
                events.append(ConveyorEvent("removed", part.part_id, part.class_name, frame_idx, part.x))
            else:
                survivors.append(part)
        self.parts = survivors

        return events

    # ------------------------------------------------------------------
    def _match(self, detections: list[Detection], frame_idx: int) -> set[int]:
        """Жадно сопоставляет детекции с экземплярами; возвращает обновлённые экземпляры.

        Детекция может сопоставиться с экземпляром того же класса, если она
        не правее его более чем на right_tolerance и не левее более чем на
        match_gate. Из кандидатов выбираются пары с минимальным расстоянием.
        """
        cfg = self.config
        self._matched_detections: set[int] = set()
        matched_parts: "_IdentitySet" = _IdentitySet()

        candidates: list[tuple[float, PartInstance, Detection]] = []
        for part in self.parts:
            for det in detections:
                if det.class_name and part.class_name and det.class_name != part.class_name:
                    continue
                dx = det.x - part.x
                if dx > cfg.right_tolerance:  # детекция правее — пролаг старой позиции
                    continue
                if dx < -cfg.match_gate:  # слишком далеко влево — не наша деталь
                    continue
                candidates.append((abs(dx), part, det))

        used_dets: set[int] = set()
        for _, part, det in sorted(candidates, key=lambda c: c[0]):
            if part in matched_parts or id(det) in used_dets:
                continue
            matched_parts.add(part)
            used_dets.add(id(det))
            self._apply_match(part, det, frame_idx)

        self._matched_detections = used_dets
        return matched_parts

    def _apply_match(self, part: PartInstance, det: Detection, frame_idx: int) -> None:
        cfg = self.config
        if det.x < part.x:
            step = det.x - part.x  # наблюдаемое смещение за тик (<= 0)
            alpha = cfg.velocity_smoothing
            new_velocity = (1 - alpha) * part.velocity + alpha * step
            part.velocity = min(_MAX_VELOCITY, max(_MIN_VELOCITY, new_velocity))
            part.x = det.x
            # Обновляем общую оценку скорости ленты.
            if self._belt_velocity is None:
                self._belt_velocity = part.velocity
            else:
                self._belt_velocity = 0.9 * self._belt_velocity + 0.1 * part.velocity
        # Если det.x чуть правее (в пределах допуска) — позицию не двигаем:
        # деталь не может ехать вправо.
        part.y = det.y
        part.last_box = (det.w, det.h)
        part.last_seen_frame = frame_idx
        part.hits += 1


class _IdentitySet:
    """Множество по идентичности объектов (dataclass-ы нехешируемы по значению)."""

    def __init__(self) -> None:
        self._ids: set[int] = set()

    def add(self, obj: object) -> None:
        self._ids.add(id(obj))

    def __contains__(self, obj: object) -> bool:
        return id(obj) in self._ids
