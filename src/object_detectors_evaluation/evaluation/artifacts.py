from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from object_detectors_evaluation.datasets.types import DatasetName
from object_detectors_evaluation.evaluation.types import DetectionBoxFormat
from object_detectors_evaluation.inference.types import ImageId
from object_detectors_evaluation.loggers import logger


class DetectionClassArtifact(BaseModel):
    """Serialized dataset class entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: int = Field(ge=0, description="Contiguous evaluator class label.")
    name: str = Field(min_length=1, description="Human-readable class name.")
    source_id: int | str = Field(description="Dataset-native class identifier.")


class DetectionClassMapArtifact(BaseModel):
    """Serialized dataset class map."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_name: DatasetName = Field(description="Dataset implementation name.")
    class_space: str = Field(min_length=1, description="Class space represented by the labels.")
    classes: tuple[DetectionClassArtifact, ...] = Field(description="Classes in evaluator label order.")


class DetectionTargetArtifact(BaseModel):
    """Serialized target record for one evaluated image."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_index: int = Field(ge=0, description="Dataset sample index.")
    box_format: DetectionBoxFormat = Field(description="Coordinate format used by `boxes`.")
    boxes: tuple[tuple[float, float, float, float], ...] = Field(description="Target boxes in evaluator coordinates.")
    boxes_org: tuple[tuple[float, float, float, float], ...] = Field(description="Original target boxes.")
    boxes_source: tuple[tuple[float, float, float, float], ...] = Field(description="Dataset-native target boxes.")
    labels: tuple[int, ...] = Field(description="Contiguous evaluator labels.")
    labels_names: tuple[str, ...] = Field(description="Human-readable target class names.")
    labels_source_ids: tuple[int | str, ...] = Field(description="Dataset-native target class identifiers.")
    labels_names_ids: tuple[int | str, ...] = Field(description="Legacy dataset-native target class identifiers.")
    iscrowd: tuple[int, ...] = Field(description="COCO-compatible crowd flags.")
    area: tuple[float, ...] | None = Field(default=None, description="Optional target areas in pixels.")
    is_occluded: tuple[int, ...] | None = Field(default=None, description="Optional Open Images occlusion flags.")
    is_truncated: tuple[int, ...] | None = Field(default=None, description="Optional Open Images truncation flags.")
    is_group_of: tuple[int, ...] | None = Field(default=None, description="Optional Open Images group-of flags.")
    is_depiction: tuple[int, ...] | None = Field(default=None, description="Optional Open Images depiction flags.")
    is_inside: tuple[int, ...] | None = Field(default=None, description="Optional Open Images inside flags.")
    image_size: tuple[int, int] = Field(description="Image size as `(height, width)`.")
    image_width: int = Field(gt=0, description="Image width in pixels.")
    image_height: int = Field(gt=0, description="Image height in pixels.")
    image_id: ImageId = Field(description="Dataset image identifier.")

    @model_validator(mode="after")
    def validate_target_lengths(self) -> Self:
        """Validate fields aligned with target boxes.

        :return: Validated target artifact.
        """
        num_targets = len(self.boxes)
        aligned_fields = {
            "boxes_org": self.boxes_org,
            "boxes_source": self.boxes_source,
            "labels": self.labels,
            "labels_names": self.labels_names,
            "labels_source_ids": self.labels_source_ids,
            "labels_names_ids": self.labels_names_ids,
        }
        for field_name, values in aligned_fields.items():
            if len(values) != num_targets:
                msg = f"Target artifact field `{field_name}` has `{len(values)}` values for `{num_targets}` boxes"
                logger.error(msg)
                raise ValueError(msg)
        if len(self.iscrowd) not in {0, num_targets}:
            msg = f"Target artifact field `iscrowd` has `{len(self.iscrowd)}` values for `{num_targets}` boxes"
            logger.error(msg)
            raise ValueError(msg)
        return self


class DetectionPredictionArtifact(BaseModel):
    """Serialized prediction record for one evaluated image."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_index: int = Field(ge=0, description="Dataset sample index.")
    box_format: DetectionBoxFormat = Field(description="Coordinate format used by `boxes`.")
    boxes: tuple[tuple[float, float, float, float], ...] = Field(description="Predicted boxes.")
    scores: tuple[float, ...] = Field(description="Prediction confidence scores.")
    labels: tuple[int, ...] = Field(description="Model-native prediction labels.")
    labels_names: tuple[str, ...] | None = Field(default=None, description="Optional prediction class names.")
    image_id: ImageId | None = Field(default=None, description="Optional dataset image identifier.")
    image_size: tuple[int, int] | None = Field(default=None, description="Optional image size as `(height, width)`.")
    class_space: str | None = Field(default=None, description="Optional model prediction class space.")

    @model_validator(mode="after")
    def validate_prediction_lengths(self) -> Self:
        """Validate fields aligned with prediction boxes.

        :return: Validated prediction artifact.
        """
        num_predictions = len(self.boxes)
        if len(self.scores) != num_predictions or len(self.labels) != num_predictions:
            msg = (
                "Prediction artifact scores and labels must align with boxes, "
                f"got `{num_predictions}` boxes, `{len(self.scores)}` scores, and `{len(self.labels)}` labels"
            )
            logger.error(msg)
            raise ValueError(msg)
        if self.labels_names is not None and len(self.labels_names) != num_predictions:
            msg = f"Prediction artifact has `{len(self.labels_names)}` class names for `{num_predictions}` boxes"
            logger.error(msg)
            raise ValueError(msg)
        return self


class DetectionLatencyArtifact(BaseModel):
    """Serialized latency record for one measured inference batch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_index: int = Field(ge=0, description="Zero-based measured batch index.")
    sample_indices: tuple[int, ...] = Field(description="Dataset sample indices in the batch.")
    image_ids: tuple[ImageId, ...] = Field(description="Dataset image identifiers in the batch.")
    batch_size: int = Field(gt=0, description="Number of images in the batch.")
    preprocess_ms: float = Field(ge=0, description="Batch preprocessing latency in milliseconds.")
    inference_ms: float = Field(ge=0, description="Batch model inference latency in milliseconds.")
    postprocess_ms: float = Field(ge=0, description="Batch postprocessing latency in milliseconds.")
    total_ms: float = Field(ge=0, description="Total measured batch latency in milliseconds.")

    @model_validator(mode="after")
    def validate_batch_lengths(self) -> Self:
        """Validate batch metadata lengths.

        :return: Validated latency artifact.
        """
        if len(self.sample_indices) != self.batch_size or len(self.image_ids) != self.batch_size:
            msg = (
                f"Latency artifact batch size is `{self.batch_size}`, with `{len(self.sample_indices)}` sample indices "
                f"and `{len(self.image_ids)}` image ids"
            )
            logger.error(msg)
            raise ValueError(msg)
        return self
