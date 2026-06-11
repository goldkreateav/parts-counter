"""Загрузка и валидация конфигурации сервиса из YAML."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _from_dict(cls, data: dict[str, Any]):
    """Создаёт dataclass из словаря, игнорируя неизвестные ключи."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class FramesConfig:
    stride: int = 5
    dedup_threshold: float = 4.0
    max_frames: int = 2000


@dataclass
class AutolabelConfig:
    model: str = "yolov8l-worldv2.pt"
    confidence: float = 0.25
    min_box_area: float = 0.01
    max_box_area: float = 0.95
    max_detections_per_frame: int = 3
    val_split: float = 0.2


@dataclass
class TrainConfig:
    model: str = "yolo11n.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    patience: int = 20
    device: str = ""


@dataclass
class ConveyorConfig:
    spawn_zone_x: float = 0.72
    count_line_x: float = 0.18
    match_gate: float = 0.18
    right_tolerance: float = 0.04
    min_spawn_separation: float = 0.25
    max_missed_frames: int = 60
    max_coast_frames: int = 25
    velocity_smoothing: float = 0.3
    default_velocity: float = -0.004


@dataclass
class CountingConfig:
    min_confidence: float = 0.35
    db_path: str = "data/counts.sqlite3"


@dataclass
class AppConfig:
    classes: dict[str, list[str]] = field(default_factory=dict)  # имя класса -> промпты
    frames: FramesConfig = field(default_factory=FramesConfig)
    autolabel: AutolabelConfig = field(default_factory=AutolabelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    conveyor: ConveyorConfig = field(default_factory=ConveyorConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)

    @property
    def class_names(self) -> list[str]:
        return list(self.classes.keys())


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    classes: dict[str, list[str]] = {}
    for name, spec in (raw.get("classes") or {}).items():
        if isinstance(spec, dict):
            classes[name] = list(spec.get("prompts") or [name])
        elif isinstance(spec, list):
            classes[name] = list(spec)
        else:
            classes[name] = [name]

    return AppConfig(
        classes=classes,
        frames=_from_dict(FramesConfig, raw.get("frames") or {}),
        autolabel=_from_dict(AutolabelConfig, raw.get("autolabel") or {}),
        train=_from_dict(TrainConfig, raw.get("train") or {}),
        conveyor=_from_dict(ConveyorConfig, raw.get("conveyor") or {}),
        counting=_from_dict(CountingConfig, raw.get("counting") or {}),
    )
