from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from object_detectors_evaluation.datasets.types import DatasetSplit


class DetectionDatasetConfig(BaseModel):
    """Configuration shared by detection datasets.

    :param dataset_dirpath: Root directory of the downloaded dataset.
    :param split: Dataset split to load.
    :param classes_of_interest: Optional class names to keep.
    :param include_crowd: Whether to keep crowd or group-of annotations in returned targets.
    :param drop_images_with_crowd: Whether to remove images that contain crowd or group-of annotations.
    :param remove_empty_images: Whether to remove images with no remaining annotations after filtering.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_dirpath: Path = Field(description="Root directory of the downloaded dataset.")
    split: DatasetSplit = Field(description="Dataset split to load.")
    classes_of_interest: list[str] | None = Field(
        default=None,
        description="Optional class names to keep.",
    )
    include_crowd: bool = Field(
        default=True,
        description="Whether to keep crowd or group-of annotations in returned targets.",
    )
    drop_images_with_crowd: bool = Field(
        default=False,
        description="Whether to remove images that contain crowd or group-of annotations.",
    )
    remove_empty_images: bool = Field(
        default=False,
        description="Whether to remove images with no remaining annotations after filtering.",
    )
