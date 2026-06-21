from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Hashable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.tv_tensors
from distinctipy import distinctipy
from natsort import natsorted
from PIL import Image
from torch import Tensor
from torchvision.datasets.vision import VisionDataset

from object_detectors_evaluation.datasets.configs import DetectionDatasetConfig
from object_detectors_evaluation.datasets.types import DetectionTarget
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.visualization.style import configure_matplotlib


def detection_collate_fn(batch: list[tuple[Any, DetectionTarget]]) -> tuple[list[Any], list[DetectionTarget]]:
    """Collate detection samples into image and target lists.

    :param batch: Sequence of ``(image, target)`` samples returned by a detection dataset.
    :return: Pair of image list and target list suitable for variable-size detection targets.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


@dataclass(frozen=True, slots=True)
class DetectionClassMap:
    """Map dataset-native class identifiers to contiguous detection labels.

    :param source_ids: Class identifiers from the source dataset, such as COCO category ids or Open Images MIDs.
    :param names: Human-readable class names aligned with ``source_ids``.
    """

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
    """Base class for object detection datasets backed by local image and annotation files.

    :param config: Detection dataset configuration.
    :param transforms: Optional callable applied jointly to image and target by TorchVision.
    :param transform: Optional image-only transform passed to ``VisionDataset``.
    :param target_transform: Optional target-only transform passed to ``VisionDataset``.
    """

    collate_fn: Callable | None = staticmethod(detection_collate_fn)

    def __init__(
        self,
        config: DetectionDatasetConfig,
        transforms: Callable | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        self.config = config
        self.dataset_dirpath = config.dataset_dirpath
        logger.info(f"Initializing dataset `{self.__class__.__name__}` with config: {config.model_dump_json(indent=4)}")
        super().__init__(
            root=str(self.dataset_dirpath),
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )
        self.split = config.split
        self.name = f"{self.dataset_dirpath.name}-{config.split}"
        self.classes_of_interest = config.classes_of_interest
        self.include_crowd = config.include_crowd
        self.drop_images_with_crowd = config.drop_images_with_crowd
        self.remove_empty_images = config.remove_empty_images
        self.class_map: DetectionClassMap | None = None

    def set_class_map(self, class_map: DetectionClassMap) -> None:
        """Set class metadata and derived lookup dictionaries.

        :param class_map: Class mapping for the concrete dataset.
        """
        self.class_map = class_map
        self.classes_names = list(class_map.names)
        self.classes_ids = list(class_map.labels)
        self.classes_source_ids = list(class_map.source_ids)
        self.classes_str2int = dict(class_map.name_to_label)
        self.classes_int2str = dict(class_map.label_to_name)
        self.classes_source_id2int = dict(class_map.source_id_to_label)
        self.classes_int2source_id = dict(class_map.label_to_source_id)
        self.num_classes = len(class_map.names)
        self.class_colors = distinctipy.get_colors(self.num_classes, rng=42)
        self.classes_int2color = dict(zip(self.classes_ids, self.class_colors))

    def get_class_names(self) -> list[str]:
        """Return human-readable class names.

        :return: Class names in internal label order.
        """
        self._require_class_map()
        return list(self.class_map.names)

    def get_class_ids(self) -> list[int]:
        """Return contiguous internal class ids.

        :return: Internal class ids in label order.
        """
        self._require_class_map()
        return list(self.class_map.labels)

    def get_source_class_ids(self) -> list[Hashable]:
        """Return dataset-native class ids.

        :return: Source class identifiers aligned with internal labels.
        """
        self._require_class_map()
        return list(self.class_map.source_ids)

    def get_raw_sample(self, index: int) -> tuple[np.ndarray, DetectionTarget]:
        """Load a raw image and target without TorchVision tensor wrapping.

        :param index: Dataset sample index.
        :return: RGB image array and target dictionary.
        """
        msg = "Subclasses must implement `get_raw_sample`"
        logger.error(msg)
        raise NotImplementedError(msg)

    def __getitem__(self, index: int) -> tuple[Tensor | Image.Image, DetectionTarget]:
        """Load a sample and apply configured TorchVision transforms.

        :param index: Dataset sample index.
        :return: Transformed image and target.
        """
        image, target = self.get_raw_sample(index)
        image = Image.fromarray(image)
        target = self._to_torch_target(target)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def plot_images_bbox(self, idxs: list[int], after_transforms: bool = False) -> plt.Figure:
        """Visualize images with bounding boxes and class labels.

        :param idxs: Dataset sample indexes to plot.
        :param after_transforms: Whether to plot samples after applying dataset transforms.
        :return: Matplotlib figure containing the plotted samples.
        """
        configure_matplotlib()
        n_images = len(idxs)
        fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
        if n_images == 1:
            axes = [axes]

        for axis, idx in zip(axes, idxs):
            if after_transforms:
                image, target = self[idx]
                image_array = self._image_to_numpy(image)
                boxes = target["boxes"].cpu().numpy()
                labels = target["labels"].cpu().numpy()
            else:
                image_array, target = self.get_raw_sample(idx)
                boxes = target["boxes"]
                labels = target["labels"]

            height, width = image_array.shape[:2]
            image_id = target["image_id"]
            axis.imshow(image_array)
            axis.set_title(f"Image ID: {image_id}\nH: {height}, W: {width}, Num of objects: {len(boxes)}")
            axis.axis("off")

            for box, label in zip(boxes, labels):
                label_int = int(label)
                x1, y1, x2, y2 = box
                color = self.classes_int2color[label_int]
                rect = plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor=color,
                    linewidth=2,
                )
                axis.add_patch(rect)
                class_name = self.classes_int2str.get(label_int, f"Class {label_int}")
                axis.text(x1, y1 - 5, class_name, color="white", bbox={"facecolor": color, "alpha": 0.7})

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        return fig

    def _to_torch_target(self, target: DetectionTarget) -> DetectionTarget:
        target = dict(target)
        target["boxes"] = torchvision.tv_tensors.BoundingBoxes(
            np.asarray(target["boxes"], dtype=np.float32),
            format="XYXY",
            canvas_size=target["image_size"],
        )
        target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
        return target

    @staticmethod
    def _image_to_numpy(image: Tensor | Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, Tensor):
            image = image.detach().cpu()
            if image.ndim == 3 and image.shape[0] in {1, 3}:
                image = image.permute(1, 2, 0)
            return image.numpy()
        return np.asarray(image)

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
    def _empty_boxes() -> np.ndarray:
        return np.zeros((0, 4), dtype=np.float32)

    @staticmethod
    def _image_filepaths(images_dirpath: Path) -> list[Path]:
        extensions = ("*.jpg", "*.jpeg", "*.png")
        filepaths: list[Path] = []
        for extension in extensions:
            filepaths.extend(images_dirpath.glob(extension))
        return natsorted(filepaths)
