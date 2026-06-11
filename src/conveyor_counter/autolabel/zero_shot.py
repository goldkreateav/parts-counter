"""Zero-shot разметка кадров через YOLO-World (открытый словарь по текстовым промптам).

Каждому классу детали из конфига соответствует список промптов; детекции по
любому из промптов класса получают его индекс. Результат — датасет в формате
YOLO (images/ + labels/ + data.yaml), готовый для обучения YOLO11.
"""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

import cv2
import yaml

from conveyor_counter.config import AutolabelConfig

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class ZeroShotLabeler:
    def __init__(self, classes: dict[str, list[str]], cfg: AutolabelConfig):
        from ultralytics import YOLOWorld

        if not classes:
            raise ValueError("В конфиге не задано ни одного класса с промптами")
        self.cfg = cfg
        self.class_names = list(classes.keys())
        # Плоский список промптов и обратное отображение промпт -> индекс класса.
        self.prompts: list[str] = []
        self.prompt_to_class: list[int] = []
        for class_idx, (_, prompts) in enumerate(classes.items()):
            for prompt in prompts:
                self.prompts.append(prompt)
                self.prompt_to_class.append(class_idx)

        self.model = YOLOWorld(cfg.model)
        self.model.set_classes(self.prompts)

    # ------------------------------------------------------------------
    def label_image(self, image_path: Path) -> list[tuple[int, float, float, float, float, float]]:
        """Возвращает список (class_idx, cx, cy, w, h, conf) в нормированных координатах."""
        cfg = self.cfg
        result = self.model.predict(str(image_path), conf=cfg.confidence, verbose=False)[0]
        boxes = []
        for box in result.boxes:
            cx, cy, w, h = box.xywhn[0].tolist()
            area = w * h
            if not (cfg.min_box_area <= area <= cfg.max_box_area):
                continue
            class_idx = self.prompt_to_class[int(box.cls[0])]
            boxes.append((class_idx, cx, cy, w, h, float(box.conf[0])))
        # Если детекций больше лимита — оставляем самые уверенные.
        boxes.sort(key=lambda b: b[5], reverse=True)
        return boxes[: cfg.max_detections_per_frame]

    # ------------------------------------------------------------------
    def label_directory(
        self,
        frames_dir: str | Path,
        out_dir: str | Path,
        preview: bool = False,
        seed: int = 42,
    ) -> Path:
        """Размечает все кадры и собирает YOLO-датасет. Возвращает путь к data.yaml."""
        frames_dir = Path(frames_dir)
        out_dir = Path(out_dir)
        images = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise RuntimeError(f"В {frames_dir} нет изображений")

        rng = random.Random(seed)
        labeled: list[tuple[Path, list]] = []
        skipped = 0
        for image_path in images:
            boxes = self.label_image(image_path)
            if not boxes:
                skipped += 1
                continue  # пустые кадры (между деталями) для обучения не нужны
            labeled.append((image_path, boxes))

        if not labeled:
            raise RuntimeError(
                "Zero-shot модель не нашла ни одной детали. "
                "Попробуйте другие промпты в configs/*.yaml или снизьте autolabel.confidence"
            )
        logger.info("Размечено %d кадров, пропущено пустых: %d", len(labeled), skipped)

        rng.shuffle(labeled)
        n_val = max(1, int(len(labeled) * self.cfg.val_split)) if len(labeled) > 1 else 0
        splits = {"val": labeled[:n_val], "train": labeled[n_val:]}

        preview_dir = out_dir / "preview"
        for split, items in splits.items():
            img_dir = out_dir / "images" / split
            lbl_dir = out_dir / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for image_path, boxes in items:
                shutil.copy2(image_path, img_dir / image_path.name)
                lines = [
                    f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    for cls, cx, cy, w, h, _ in boxes
                ]
                (lbl_dir / f"{image_path.stem}.txt").write_text("\n".join(lines) + "\n")
                if preview:
                    self._save_preview(image_path, boxes, preview_dir)

        data_yaml = out_dir / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(out_dir.resolve()),
                    "train": "images/train",
                    "val": "images/val" if n_val else "images/train",
                    "names": dict(enumerate(self.class_names)),
                },
                allow_unicode=True,
                sort_keys=False,
            )
        )
        logger.info("Датасет собран: %s (train=%d, val=%d)", out_dir, len(splits["train"]), n_val)
        return data_yaml

    # ------------------------------------------------------------------
    def _save_preview(self, image_path: Path, boxes: list, preview_dir: Path) -> None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(image_path))
        if img is None:
            return
        h, w = img.shape[:2]
        for cls, cx, cy, bw, bh, conf in boxes:
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(img, f"{self.class_names[cls]} {conf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        cv2.imwrite(str(preview_dir / image_path.name), img)
