"""CLI сервиса: extract-frames -> autolabel -> train -> count -> report."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from conveyor_counter.config import load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="conveyor-counter",
        description="Zero-shot разметка, обучение YOLO11 и подсчёт деталей на конвейере",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="путь к YAML-конфигу")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract-frames", help="извлечь кадры из видео для разметки")
    p.add_argument("--video", required=True, help="путь к видео с конвейером")
    p.add_argument("--out", default="data/frames", help="каталог для кадров")

    p = sub.add_parser("autolabel", help="zero-shot разметка кадров и сборка YOLO-датасета")
    p.add_argument("--frames", default="data/frames", help="каталог с кадрами")
    p.add_argument("--out", default="data/dataset", help="каталог датасета")
    p.add_argument("--preview", action="store_true", help="сохранять превью разметки для проверки")

    p = sub.add_parser("train", help="обучить YOLO11 на собранном датасете")
    p.add_argument("--data", default="data/dataset/data.yaml", help="путь к data.yaml")
    p.add_argument("--project", default="runs", help="каталог для результатов обучения")

    p = sub.add_parser("count", help="подсчёт деталей на видео или в прямом эфире")
    p.add_argument("--source", required=True,
                   help="видеофайл, rtsp://-URL или индекс камеры (например, 0)")
    p.add_argument("--weights", required=True, help="веса YOLO11 (best.pt)")
    p.add_argument("--show", action="store_true", help="показывать окно с визуализацией")
    p.add_argument("--save", default=None, help="сохранить аннотированное видео в файл")

    p = sub.add_parser("report", help="показать счётчики деталей по дням")
    p.add_argument("--day", default=None, help="день в формате YYYY-MM-DD (по умолчанию — все)")

    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "extract-frames":
        from conveyor_counter.autolabel.frames import extract_frames

        extract_frames(args.video, args.out, config.frames)

    elif args.command == "autolabel":
        from conveyor_counter.autolabel.zero_shot import ZeroShotLabeler

        labeler = ZeroShotLabeler(config.classes, config.autolabel)
        data_yaml = labeler.label_directory(args.frames, args.out, preview=args.preview)
        print(f"Датасет готов: {data_yaml}")

    elif args.command == "train":
        from conveyor_counter.training.train import train_yolo11

        best = train_yolo11(args.data, config.train, project=args.project)
        print(f"Лучшие веса: {best}")

    elif args.command == "count":
        from conveyor_counter.counting.service import CounterService

        service = CounterService(
            weights=args.weights,
            config=config,
            source=args.source,
            show=args.show,
            save_path=args.save,
        )
        counts = service.run()
        print("Счётчики за сегодня:")
        for name, count in sorted(counts.items()):
            print(f"  {name}: {count}")

    elif args.command == "report":
        from conveyor_counter.counting.store import CountStore

        store = CountStore(config.counting.db_path)
        if args.day:
            counts = store.get_counts(date.fromisoformat(args.day))
            for name, count in sorted(counts.items()):
                print(f"{args.day}  {name}: {count}")
            if not counts:
                print(f"За {args.day} записей нет")
        else:
            rows = store.all_days()
            if not rows:
                print("Записей нет")
            for day, name, count in rows:
                print(f"{day}  {name}: {count}")
        store.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
