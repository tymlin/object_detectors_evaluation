from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import fiftyone as fo
import fiftyone.zoo as foz

from object_detectors_evaluation.consts import FIFTYONE_DATASETS_DIRPATH
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.utils.types import Split

COCO_2017_DATASET_NAME = "coco-2017"
OPEN_IMAGES_V7_DATASET_NAME = "open-images-v7"


def configure_fiftyone_dataset_dir(dataset_dirpath: str | Path) -> Path:
    dataset_dirpath = Path(dataset_dirpath)
    dataset_dirpath.mkdir(parents=True, exist_ok=True)
    fo.config.dataset_zoo_dir = str(dataset_dirpath)
    fo.config.default_dataset_dir = str(dataset_dirpath)
    logger.info(f"Configured FiftyOne dataset directory at path: '{dataset_dirpath}'")
    return dataset_dirpath


def download_coco_dataset(
    dataset_dirpath: str | Path | None = FIFTYONE_DATASETS_DIRPATH,
    split: Split = "validation",
    classes: Sequence[str] | None = None,
    max_samples: int | None = None,
    label_types: Sequence[str] = ("detections",),
    **kwargs: Any,
) -> Any:
    return download_zoo_detection_dataset(
        dataset_name=COCO_2017_DATASET_NAME,
        dataset_dirpath=dataset_dirpath,
        split=split,
        classes=classes,
        max_samples=max_samples,
        label_types=label_types,
        **kwargs,
    )


def download_open_images_dataset(
    dataset_dirpath: str | Path | None = FIFTYONE_DATASETS_DIRPATH,
    split: Split = "test",
    classes: Sequence[str] | None = None,
    max_samples: int | None = None,
    label_types: Sequence[str] = ("detections",),
    **kwargs: Any,
) -> Any:
    return download_zoo_detection_dataset(
        dataset_name=OPEN_IMAGES_V7_DATASET_NAME,
        dataset_dirpath=dataset_dirpath,
        split=split,
        classes=classes,
        max_samples=max_samples,
        label_types=label_types,
        **kwargs,
    )


def download_zoo_detection_dataset(
    dataset_name: str,
    dataset_dirpath: str | Path | None,
    split: Split,
    classes: Sequence[str] | None = None,
    max_samples: int | None = None,
    label_types: Sequence[str] = ("detections",),
    **kwargs: Any,
) -> Any:
    if dataset_dirpath is not None:
        configure_fiftyone_dataset_dir(dataset_dirpath)

    logger.info(f"Downloading FiftyOne dataset `{dataset_name}` split `{split}`")
    dataset = foz.download_zoo_dataset(
        dataset_name,
        split=split,
        label_types=list(label_types),
        classes=list(classes) if classes is not None else None,
        max_samples=max_samples,
        **kwargs,
    )
    logger.info(f"Downloaded FiftyOne dataset `{dataset_name}` split `{split}`")
    return dataset
