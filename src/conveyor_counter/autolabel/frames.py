"""Извлечение кадров из видео для разметки.

Берёт каждый N-й кадр и отбрасывает почти не изменившиеся кадры
(пролаги/зависания видео), чтобы датасет не забивался дубликатами.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from conveyor_counter.config import FramesConfig

logger = logging.getLogger(__name__)


def extract_frames(video_path: str | Path, out_dir: str | Path, cfg: FramesConfig) -> int:
    """Сохраняет кадры в out_dir как JPEG. Возвращает количество сохранённых кадров."""
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    saved = 0
    frame_idx = 0
    prev_gray: np.ndarray | None = None
    stem = video_path.stem

    try:
        while saved < cfg.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % cfg.stride != 0:
                frame_idx += 1
                continue

            gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
                if diff < cfg.dedup_threshold:
                    frame_idx += 1
                    continue  # кадр-дубликат (пролаг видео) — пропускаем
            prev_gray = gray

            out_path = out_dir / f"{stem}_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            frame_idx += 1
    finally:
        cap.release()

    logger.info("Сохранено %d кадров из %s в %s", saved, video_path, out_dir)
    return saved
