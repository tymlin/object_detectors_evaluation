from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Hashable

import numpy as np
import torch
import torchvision.tv_tensors
from natsort import natsorted
from PIL import Image
from torch import Tensor
from torchvision.datasets.vision import VisionDataset

from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.utils.types import Split

DetectionTarget = dict[str, Any]


def detection_collate_fn(batch: list[tuple[Any, DetectionTarget]]) -> tuple[list[Any], list[DetectionTarget]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


@dataclass(frozen=True, slots=True)
class DetectionClassMap:
    source_ids: tuple[Hashable, ...]
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.source_ids) != len(self.names):
            msg = "source_ids and names must have the same length"
            logger.error(msg)
            raise ValueError(msg)
        if len(set(self.source_ids)) != len(self.source_ids):
            msg = "source_ids must be unique"
            logger.error(msg)
            raise ValueError(msg)
        if len(set(self.names)) != len(self.names):
            msg = "names must be unique"
            logger.error(msg)
            raise ValueError(msg)

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(range(len(self.names)))

    @property
    def source_id_to_label(self) -> dict[Hashable, int]:
        return {source_id: label for label, source_id in enumerate(self.source_ids)}

    @property
    def label_to_source_id(self) -> dict[int, Hashable]:
        return {label: source_id for label, source_id in enumerate(self.source_ids)}

    @property
    def name_to_label(self) -> dict[str, int]:
        return {name: label for label, name in enumerate(self.names)}

    @property
    def label_to_name(self) -> dict[int, str]:
        return {label: name for label, name in enumerate(self.names)}

    @property
    def source_id_to_name(self) -> dict[Hashable, str]:
        return dict(zip(self.source_ids, self.names))

    def label_for_source_id(self, source_id: Hashable) -> int:
        return self.source_id_to_label[source_id]

    def name_for_source_id(self, source_id: Hashable) -> str:
        return self.source_id_to_name[source_id]

    def source_ids_for_names(self, names: list[str] | tuple[str, ...]) -> set[Hashable]:
        unknown = natsorted(set(names) - set(self.names))
        if unknown:
            msg = f"Unknown class names: {unknown}"
            logger.error(msg)
            raise ValueError(msg)
        name_to_source_id = {name: source_id for source_id, name in zip(self.source_ids, self.names)}
        return {name_to_source_id[name] for name in names}


class BaseDetectionDataset(VisionDataset):
    collate_fn: Callable | None = staticmethod(detection_collate_fn)

    def __init__(
        self,
        dataset_dirpath: str | Path,
        split: Split,
        transforms: Callable | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        classes_of_interest: list[str] | None = None,
    ) -> None:
        self.dataset_dirpath = Path(dataset_dirpath)
        super().__init__(
            root=str(self.dataset_dirpath),
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self.split = split
        self.name = f"{self.dataset_dirpath.name}-{split}"
        self.classes_of_interest = classes_of_interest
        self.class_map: DetectionClassMap | None = None

    def set_class_map(self, class_map: DetectionClassMap) -> None:
        self.class_map = class_map
        self.classes_names = list(class_map.names)
        self.classes_ids = list(class_map.labels)
        self.classes_source_ids = list(class_map.source_ids)
        self.classes_str2int = dict(class_map.name_to_label)
        self.classes_int2str = dict(class_map.label_to_name)
        self.classes_source_id2int = dict(class_map.source_id_to_label)
        self.classes_int2source_id = dict(class_map.label_to_source_id)
        self.num_classes = len(class_map.names)

    def get_class_names(self) -> list[str]:
        self._require_class_map()
        return list(self.class_map.names)

    def get_class_ids(self) -> list[int]:
        self._require_class_map()
        return list(self.class_map.labels)

    def get_source_class_ids(self) -> list[Hashable]:
        self._require_class_map()
        return list(self.class_map.source_ids)

    def get_raw_sample(self, index: int) -> tuple[np.ndarray, DetectionTarget]:
        msg = "Subclasses must implement `get_raw_sample`"
        logger.error(msg)
        raise NotImplementedError(msg)

    def __getitem__(self, index: int) -> tuple[Tensor | Image.Image, DetectionTarget]:
        image, target = self.get_raw_sample(index)
        image = Image.fromarray(image)
        target = self._to_torch_target(target)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _to_torch_target(self, target: DetectionTarget) -> DetectionTarget:
        target = dict(target)
        target["boxes"] = torchvision.tv_tensors.BoundingBoxes(
            np.asarray(target["boxes"], dtype=np.float32),
            format="XYXY",
            canvas_size=target["image_size"],
        )
        target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
        return target

    def _require_class_map(self) -> None:
        if self.class_map is None:
            msg = "Dataset class map has not been initialized"
            logger.error(msg)
            raise RuntimeError(msg)

    def _source_id_filter(self) -> set[Hashable] | None:
        self._require_class_map()
        if self.classes_of_interest is None:
            return None
        return self.class_map.source_ids_for_names(self.classes_of_interest)

    @staticmethod
    def _load_json(filepath: Path) -> dict[str, Any]:
        if not filepath.exists():
            msg = f"Could not find JSON file at path: '{filepath}'"
            logger.error(msg)
            raise FileNotFoundError(msg)
        with filepath.open() as file:
            return json.load(file)

    @staticmethod
    def _load_optional_json(filepath: Path) -> dict[str, Any]:
        if not filepath.exists():
            return {}
        return BaseDetectionDataset._load_json(filepath)

    @staticmethod
    def _empty_boxes() -> np.ndarray:
        return np.zeros((0, 4), dtype=np.float32)

    @staticmethod
    def _image_filepaths(images_dirpath: Path) -> list[Path]:
        extensions = ("*.jpg", "*.jpeg", "*.png")
        filepaths: list[Path] = []
        for extension in extensions:
            filepaths.extend(images_dirpath.glob(extension))
        return natsorted(filepaths)
