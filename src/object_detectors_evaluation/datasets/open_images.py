from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image

from object_detectors_evaluation.datasets.configs import DetectionDatasetConfig

from .base import (
    BaseDetectionDataset,
    DetectionClassMap,
    DetectionTarget,
)


class OpenImagesDataset(BaseDetectionDataset):
    """Open Images detection dataset reader for FiftyOne-style exported data.

    Expected layout is ``<dataset_dirpath>/<split>/data`` for images,
    ``<dataset_dirpath>/<split>/labels/detections.csv`` for annotations, and
    ``<dataset_dirpath>/<split>/metadata/classes.csv`` for class metadata.
    Open Images ``IsGroupOf`` annotations are treated as the dataset's crowd equivalent.

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
        self.annotations_filepath = self.dataset_dirpath / self.split / "labels" / "detections.csv"
        self.classes_filepath = self.dataset_dirpath / self.split / "metadata" / "classes.csv"
        self.info_filepath = self.dataset_dirpath / "info.json"
        self.images_filepaths = self._image_filepaths(self.images_dirpath)

        self.info = self._load_optional_json(self.info_filepath)
        self.classes = pd.read_csv(self.classes_filepath, header=None, names=["source_id", "name"])
        self.annotations = pd.read_csv(self.annotations_filepath)

        self.set_class_map(
            DetectionClassMap(
                source_ids=tuple(self.classes["source_id"].tolist()),
                names=tuple(self.classes["name"].tolist()),
            )
        )
        class_map = self.class_map
        self.classes_strid2str = dict(class_map.source_id_to_name)
        self.classes_str2strid = {name: source_id for source_id, name in zip(class_map.source_ids, class_map.names)}
        self.classes_strid2int = dict(class_map.source_id_to_label)
        self.classes_int2strid = dict(class_map.label_to_source_id)

        self.source_id_filter = self._source_id_filter()
        self.annotations_by_image_id = {
            image_id: rows.copy() for image_id, rows in self.annotations.groupby("ImageID", sort=False)
        }
        self.images_filepaths = self._filter_image_filepaths(self.images_filepaths)

    def get_raw_sample(self, index: int) -> tuple[np.ndarray, DetectionTarget]:
        """Load an Open Images sample before TorchVision transforms are applied.

        :param index: Dataset sample index.
        :return: RGB image array and detection target.
        """
        image_filepath = self.images_filepaths[index]
        image_id = image_filepath.stem
        image = np.asarray(Image.open(image_filepath).convert("RGB"))
        image_height, image_width = image.shape[:2]
        image_size = (image_height, image_width)

        annotations = self._filter_annotations(self.annotations_by_image_id.get(image_id, self.annotations.iloc[0:0]))

        labels_source_ids = annotations["LabelName"].tolist()
        labels_names = [self.class_map.name_for_source_id(source_id) for source_id in labels_source_ids]
        labels = [self.class_map.label_for_source_id(source_id) for source_id in labels_source_ids]

        boxes_normalized = annotations[["XMin", "YMin", "XMax", "YMax"]].to_numpy(dtype=np.float32)
        if len(boxes_normalized) == 0:
            boxes_normalized = self._empty_boxes()
        boxes = boxes_normalized * np.asarray(
            [image_width, image_height, image_width, image_height],
            dtype=np.float32,
        )

        target: DetectionTarget = {
            "boxes": boxes,
            "boxes_org": boxes_normalized.copy(),
            "boxes_source": boxes_normalized,
            "labels": labels,
            "labels_names": labels_names,
            "labels_source_ids": labels_source_ids,
            "labels_names_ids": labels_source_ids,
            "is_occluded": self._column_values(annotations, "IsOccluded"),
            "is_truncated": self._column_values(annotations, "IsTruncated"),
            "is_group_of": self._column_values(annotations, "IsGroupOf"),
            "iscrowd": self._column_values(annotations, "IsGroupOf"),
            "is_depiction": self._column_values(annotations, "IsDepiction"),
            "is_inside": self._column_values(annotations, "IsInside"),
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
    def _column_values(annotations: pd.DataFrame, column: str) -> list:
        if column not in annotations.columns:
            return []
        return annotations[column].tolist()

    def _filter_annotations(self, annotations: pd.DataFrame) -> pd.DataFrame:
        """Filter Open Images annotations by class and crowd policy.

        :param annotations: Raw Open Images annotations for one image.
        :return: Filtered annotations.
        """
        annotations = annotations.copy()
        if self.source_id_filter is not None:
            annotations = annotations[annotations["LabelName"].isin(self.source_id_filter)]
        if not self.include_crowd and "IsGroupOf" in annotations.columns:
            annotations = annotations[annotations["IsGroupOf"] == 0]
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
            annotations = self.annotations_by_image_id.get(image_filepath.stem, self.annotations.iloc[0:0]).copy()
            if self.source_id_filter is not None:
                annotations = annotations[annotations["LabelName"].isin(self.source_id_filter)]
            has_crowd = "IsGroupOf" in annotations.columns and bool((annotations["IsGroupOf"] == 1).any())
            if self.drop_images_with_crowd and has_crowd:
                continue
            if self.remove_empty_images and len(self._filter_annotations(annotations)) == 0:
                continue
            filtered_image_filepaths.append(image_filepath)

        return filtered_image_filepaths
