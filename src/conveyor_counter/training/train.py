"""Обучение YOLO11 на датасете, собранном zero-shot разметкой."""

from __future__ import annotations

import logging
from pathlib import Path

from conveyor_counter.config import TrainConfig

logger = logging.getLogger(__name__)


def train_yolo11(data_yaml: str | Path, cfg: TrainConfig, project: str = "runs") -> Path:
    """Запускает обучение и возвращает путь к лучшим весам (best.pt)."""
    from ultralytics import YOLO

    model = YOLO(cfg.model)
    results = model.train(
        data=str(data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        patience=cfg.patience,
        device=cfg.device or None,
        project=project,
        name="conveyor_yolo11",
        exist_ok=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("Обучение завершено, лучшие веса: %s", best)
    return best
