"""Builds small, deliberately broken datasets in the Ultralytics layout."""
from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image


class DatasetFactory:
    def __init__(self, root: Path, fmt: str):
        self.root = root
        self.fmt = fmt
        self._count = 0

    def image_path(self, name: str, split: str = "train", suffix: str = ".png") -> Path:
        return self.root / "images" / split / f"{name}{suffix}"

    def label_path(self, name: str, split: str = "train") -> Path:
        return self.root / "labels" / split / f"{name}.txt"

    def add_image(self, name: str, split: str = "train", size: tuple[int, int] = (640, 640)) -> Path:
        path = self.image_path(name, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._count += 1
        color = (self._count % 256, self._count // 256 % 256, 0)  # distinct pixels so images don't hash alike
        Image.new("RGB", size, color).save(path)
        return path

    def add_label(self, name: str, text: str, split: str = "train", size: tuple[int, int] = (640, 640)) -> Path:
        """Write a label file, creating its image first if it doesn't exist."""
        if not self.image_path(name, split).exists():
            self.add_image(name, split, size)
        path = self.label_path(name, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
        return path

    def write_data_yaml(self, names: list[str], splits: tuple[str, ...] = ("train",)) -> Path:
        path = self.root / "data.yaml"
        cfg = {split: f"images/{split}" for split in splits}
        cfg["names"] = names
        path.write_text(yaml.safe_dump(cfg))
        return path


def make_dataset(tmp_path: Path, fmt: str) -> DatasetFactory:
    return DatasetFactory(tmp_path / "dataset", fmt)
