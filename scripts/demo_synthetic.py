"""Генерация синтетических данных для сквозной проверки пайплайна без реального видео.

Создаёт:
* видео конвейера (детали-прямоугольники едут справа налево, с пропусками кадров,
  имитирующими пролаги, и одной «бракованной» деталью, исчезающей с ленты);
* YOLO-датасет с эталонной разметкой для быстрого обучения.

Использование:
    python3 scripts/demo_synthetic.py --out data/demo
    conveyor-counter train --data data/demo/dataset/data.yaml
    conveyor-counter count --source data/demo/conveyor.mp4 --weights <best.pt>
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

W, H = 640, 360
PART_W, PART_H = 150, 200
SPEED = 6  # пикселей за кадр, движение влево


def draw_part(frame: np.ndarray, cx: int, cy: int, rng: random.Random) -> None:
    """Рисует «деталь»: светлая металлическая панель с рёбрами жёсткости."""
    x1, y1 = cx - PART_W // 2, cy - PART_H // 2
    x2, y2 = cx + PART_W // 2, cy + PART_H // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (190, 195, 200), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 125, 130), 2)
    for i in range(4):  # рёбра жёсткости
        ry = y1 + 30 + i * 40
        cv2.rectangle(frame, (x1 + 15, ry), (x2 - 15, ry + 12), (150, 155, 160), -1)
    for _ in range(6):  # отверстия
        hx = rng.randint(x1 + 10, x2 - 10)
        hy = rng.randint(y1 + 10, y2 - 10)
        cv2.circle(frame, (hx, hy), 4, (40, 40, 45), -1)


def make_background() -> np.ndarray:
    bg = np.full((H, W, 3), 30, dtype=np.uint8)
    noise = np.random.default_rng(0).integers(0, 12, (H, W, 3), dtype=np.uint8)
    return cv2.add(bg, noise)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/demo", help="каталог для видео и датасета")
    parser.add_argument("--parts", type=int, default=3, help="сколько деталей проедет полностью")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)

    # Расписание: детали стартуют справа за кадром с интервалом; одна — брак,
    # исчезает на середине ленты.
    gap = (W + PART_W) // SPEED + 15
    schedule = [(i * gap, False) for i in range(args.parts)]
    schedule.append((args.parts * gap, True))  # бракованная — последняя
    total_frames = (args.parts + 1) * gap + 40

    video_path = out_dir / "conveyor.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))

    img_dir = out_dir / "dataset" / "images" / "train"
    lbl_dir = out_dir / "dataset" / "labels" / "train"
    val_img_dir = out_dir / "dataset" / "images" / "val"
    val_lbl_dir = out_dir / "dataset" / "labels" / "val"
    for d in (img_dir, lbl_dir, val_img_dir, val_lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    saved = 0
    for frame_idx in range(total_frames):
        frame = make_background().copy()
        boxes = []
        for start, defective, in schedule:
            t = frame_idx - start
            if t < 0:
                continue
            cx = W + PART_W // 2 - t * SPEED
            if cx < -PART_W // 2:
                continue
            if defective and cx < W * 0.55:
                continue  # брак: деталь убрали с ленты на середине
            cy = H // 2
            part_rng = random.Random(start)
            draw_part(frame, cx, cy, part_rng)
            x1 = max(0, cx - PART_W // 2)
            x2 = min(W, cx + PART_W // 2)
            if x2 - x1 > 20:
                bcx, bw = (x1 + x2) / 2 / W, (x2 - x1) / W
                boxes.append((0, bcx, cy / H, bw, PART_H / H))

        writer.write(frame)

        # Каждый 4-й кадр с деталью — в датасет.
        if boxes and frame_idx % 4 == 0:
            to_val = saved % 5 == 4
            idir, ldir = (val_img_dir, val_lbl_dir) if to_val else (img_dir, lbl_dir)
            name = f"synthetic_{frame_idx:05d}"
            cv2.imwrite(str(idir / f"{name}.jpg"), frame)
            (ldir / f"{name}.txt").write_text(
                "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes) + "\n"
            )
            saved += 1

    writer.release()

    data_yaml = out_dir / "dataset" / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.resolve()}/dataset\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: floor_panel\n"
    )
    print(f"Видео: {video_path} ({total_frames} кадров)")
    print(f"Датасет: {data_yaml} ({saved} изображений)")
    print(f"Ожидаемый счёт деталей: {args.parts} (+1 брак, который не считается)")


if __name__ == "__main__":
    main()
