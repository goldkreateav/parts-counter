"""Сервис подсчёта: видеопоток -> YOLO11 -> виртуальный конвейер -> счётчики."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from conveyor_counter.config import AppConfig
from conveyor_counter.counting.store import CountStore
from conveyor_counter.counting.virtual_conveyor import Detection, VirtualConveyor

logger = logging.getLogger(__name__)

_COLORS = [
    (80, 200, 120), (255, 160, 60), (90, 160, 255),
    (200, 100, 255), (60, 220, 220), (255, 90, 130),
]


class CounterService:
    """Читает видео (файл / RTSP / индекс камеры), детектит YOLO11 и считает детали."""

    def __init__(
        self,
        weights: str | Path,
        config: AppConfig,
        source: str,
        show: bool = False,
        save_path: str | Path | None = None,
    ):
        from ultralytics import YOLO  # импорт здесь, чтобы не тянуть torch без надобности

        self.model = YOLO(str(weights))
        self.config = config
        self.source = source
        self.show = show
        self.save_path = Path(save_path) if save_path else None
        self.conveyor = VirtualConveyor(config.conveyor)
        self.store = CountStore(config.counting.db_path)

    # ------------------------------------------------------------------
    def run(self) -> dict[str, int]:
        """Основной цикл. Возвращает счётчики за сегодняшний день."""
        source: str | int = self.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)  # индекс камеры для прямого эфира

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть источник видео: {self.source}")

        writer = None
        if self.save_path:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(self.save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                detections = self._detect(frame)
                events = self.conveyor.update(detections, frame_idx)
                self._handle_events(events)

                if self.show or writer is not None:
                    annotated = self._draw(frame, detections)
                    if writer is not None:
                        writer.write(annotated)
                    if self.show:
                        cv2.imshow("conveyor-counter", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                frame_idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if self.show:
                cv2.destroyAllWindows()

        counts = self.store.get_counts()
        logger.info("Итог за день: %s", counts)
        return counts

    # ------------------------------------------------------------------
    def _detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        result = self.model.predict(
            frame, conf=self.config.counting.min_confidence, verbose=False
        )[0]
        detections: list[Detection] = []
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    x=(x1 + x2) / 2 / w,
                    y=(y1 + y2) / 2 / h,
                    w=(x2 - x1) / w,
                    h=(y2 - y1) / h,
                    class_name=names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                )
            )
        return detections

    def _handle_events(self, events) -> None:
        for ev in events:
            self.store.log_event(ev.kind, ev.part_id, ev.class_name, ev.frame_idx)
            if ev.kind == "counted":
                total = self.store.increment(ev.class_name)
                logger.info(
                    "Деталь #%d (%s) пересекла линию. Всего за день: %d",
                    ev.part_id, ev.class_name, total,
                )
            elif ev.kind == "removed":
                logger.info("Деталь #%d (%s) убрана с ленты (брак/пропажа)", ev.part_id, ev.class_name)
            elif ev.kind == "spawned":
                logger.info("Новая деталь #%d (%s) появилась справа", ev.part_id, ev.class_name)

    # ------------------------------------------------------------------
    def _draw(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        cfg = self.config.conveyor
        h, w = frame.shape[:2]
        out = frame.copy()

        # Линия подсчёта и зона спавна.
        line_x = int(cfg.count_line_x * w)
        spawn_x = int(cfg.spawn_zone_x * w)
        cv2.line(out, (line_x, 0), (line_x, h), (0, 0, 255), 2)
        cv2.line(out, (spawn_x, 0), (spawn_x, h), (0, 255, 255), 1)
        cv2.putText(out, "count line", (line_x + 5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(out, "spawn zone ->", (spawn_x + 5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Сырые детекции YOLO — серым.
        for det in detections:
            x1 = int((det.x - det.w / 2) * w)
            y1 = int((det.y - det.h / 2) * h)
            x2 = int((det.x + det.w / 2) * w)
            y2 = int((det.y + det.h / 2) * h)
            cv2.rectangle(out, (x1, y1), (x2, y2), (160, 160, 160), 1)

        # Виртуальные экземпляры — цветом, с id и классом.
        for part in self.conveyor.parts:
            color = _COLORS[part.part_id % len(_COLORS)]
            cx, cy = int(part.x * w), int(part.y * h)
            bw = int(part.last_box[0] * w) or 80
            bh = int(part.last_box[1] * h) or 80
            cv2.rectangle(out, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2), color, 2)
            cv2.putText(out, f"#{part.part_id} {part.class_name}", (cx - bw // 2, cy - bh // 2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Табло счётчиков за день.
        y_offset = 55
        for name, count in sorted(self.store.get_counts().items()):
            cv2.putText(out, f"{name}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_offset += 30
        return out
