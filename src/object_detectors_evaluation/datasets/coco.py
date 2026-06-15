from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from object_detectors_evaluation.datasets.configs import DetectionDatasetConfig
from object_detectors_evaluation.datasets.types import DetectionTarget
from object_detectors_evaluation.loggers import logger

from .base import (
    BaseDetectionDataset,
    DetectionClassMap,
)


class COCODataset(BaseDetectionDataset):
    """COCO detection dataset reader for FiftyOne-style exported data.

    Expected layout is ``<dataset_dirpath>/<split>/data`` for images and
    ``<dataset_dirpath>/<split>/labels.json`` for COCO annotations.

    :param config: Detection dataset configuration.
    :param transforms: Optional callable applied jointly to image and target by TorchVision.
    :param transform: Optional image-only transform passed to ``VisionDataset``.
    :param target_transform: Optional target-only transform passed to ``VisionDataset``.
    """

    def __init__(
        self,
        config: DetectionDatasetConfig,
        transforms: Callable | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        super().__init__(
            config=config,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )

        self.images_dirpath = self.dataset_dirpath / self.split / "data"
        self.annotations_filepath = self.dataset_dirpath / self.split / "labels.json"
        self.info_filepath = self.dataset_dirpath / "info.json"
        self.images_filepaths = self._image_filepaths(self.images_dirpath)

        self.info = self._load_optional_json(self.info_filepath)
        self.coco_data = self._load_json(self.annotations_filepath)
        self.categories = self.coco_data.get("categories", [])
        self.images_info = self.coco_data.get("images", [])
        self.annotations_data = self.coco_data.get("annotations", [])

        self.set_class_map(
            DetectionClassMap(
                source_ids=tuple(category["id"] for category in self.categories),
                names=tuple(category["name"] for category in self.categories),
            )
        )
        self.classes_coco_id2int = dict(self.classes_source_id2int)
        self.classes_int2coco_id = dict(self.classes_int2source_id)

        self.image_id_to_info = {image_info["id"]: image_info for image_info in self.images_info}
        self.filename_to_image_info = {image_info["file_name"]: image_info for image_info in self.images_info}
        self.image_id_to_annotations = self._annotations_by_image_id(self.annotations_data)
        self.source_id_filter = self._source_id_filter()
        self.images_filepaths = self._filter_image_filepaths(self.images_filepaths)
        logger.info(
            f"Initialized dataset `{self.__class__.__name__}` with `{len(self)}` images, "
            f"`{self.num_classes}` classes, `{len(self.annotations_data)}` annotations, "
            f"images path: '{self.images_dirpath}', annotations path: '{self.annotations_filepath}'"
        )

    def get_raw_sample(self, index: int) -> tuple[np.ndarray, DetectionTarget]:
        """Load a COCO sample before TorchVision transforms are applied.

        :param index: Dataset sample index.
        :return: RGB image array and detection target.
        """
        image_filepath = self.images_filepaths[index]
        image_info = self.filename_to_image_info.get(image_filepath.name)
        if image_info is None:
            msg = f"Could not find COCO image metadata for `{image_filepath.name}`"
            logger.error(msg)
            raise ValueError(msg)

        image_id = image_info["id"]
        image = np.asarray(Image.open(image_filepath).convert("RGB"))
        image_height = int(image_info.get("height", image.shape[0]))
        image_width = int(image_info.get("width", image.shape[1]))
        image_size = (image_height, image_width)

        annotations = self._filter_annotations(self.image_id_to_annotations.get(image_id, []))

        boxes: list[list[float]] = []
        boxes_source: list[list[float]] = []
        labels: list[int] = []
        labels_names: list[str] = []
        labels_source_ids: list[int] = []
        iscrowd: list[int] = []
        area: list[float] = []

        for annotation in annotations:
            x, y, width, height = annotation["bbox"]
            category_id = annotation["category_id"]

            boxes.append([x, y, x + width, y + height])
            boxes_source.append([x, y, width, height])
            labels.append(self.class_map.label_for_source_id(category_id))
            labels_names.append(self.class_map.name_for_source_id(category_id))
            labels_source_ids.append(category_id)
            iscrowd.append(annotation.get("iscrowd", 0))
            area.append(annotation.get("area", width * height))

        boxes_array = np.asarray(boxes, dtype=np.float32) if boxes else self._empty_boxes()
        boxes_source_array = np.asarray(boxes_source, dtype=np.float32) if boxes_source else self._empty_boxes()

        target: DetectionTarget = {
            "boxes": boxes_array,
            "boxes_org": boxes_array.copy(),
            "boxes_source": boxes_source_array,
            "labels": labels,
            "labels_names": labels_names,
            "labels_source_ids": labels_source_ids,
            "labels_names_ids": labels_source_ids,
            "iscrowd": iscrowd,
            "area": area,
            "image_size": image_size,
            "image_width": image_width,
            "image_height": image_height,
            "image_id": image_id,
        }
        return image, target

    def __len__(self) -> int:
        """Return the number of images after configured image-level filtering.

        :return: Dataset length.
        """
        return len(self.images_filepaths)

    @staticmethod
    def _annotations_by_image_id(annotations: list[dict[str, Any]]) -> dict[Any, list[dict[str, Any]]]:
        annotations_by_image_id: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for annotation in annotations:
            annotations_by_image_id[annotation["image_id"]].append(annotation)
        return dict(annotations_by_image_id)

    def _filter_annotations(self, annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter COCO annotations by class and crowd policy.

        :param annotations: Raw COCO annotations for one image.
        :return: Filtered annotations.
        """
        if self.source_id_filter is not None:
            annotations = [
                annotation for annotation in annotations if annotation["category_id"] in self.source_id_filter
            ]
        if not self.include_crowd:
            annotations = [annotation for annotation in annotations if not self._is_crowd_annotation(annotation)]
        return annotations

    def _filter_image_filepaths(self, image_filepaths: list[Path]) -> list[Path]:
        """Filter image paths according to crowd and empty-image policy.

        :param image_filepaths: Candidate image file paths.
        :return: Filtered image file paths.
        """
        if not self.drop_images_with_crowd and not self.remove_empty_images:
            return image_filepaths

        filtered_image_filepaths = []
        for image_filepath in image_filepaths:
            image_info = self.filename_to_image_info.get(image_filepath.name)
            if image_info is None:
                continue

            annotations = self.image_id_to_annotations.get(image_info["id"], [])
            if self.source_id_filter is not None:
                annotations = [
                    annotation for annotation in annotations if annotation["category_id"] in self.source_id_filter
                ]
            has_crowd = any(self._is_crowd_annotation(annotation) for annotation in annotations)
            if self.drop_images_with_crowd and has_crowd:
                continue
            if self.remove_empty_images and len(self._filter_annotations(annotations)) == 0:
                continue
            filtered_image_filepaths.append(image_filepath)

        return filtered_image_filepaths

    @staticmethod
    def _is_crowd_annotation(annotation: dict[str, Any]) -> bool:
        return bool(annotation.get("iscrowd", 0))
