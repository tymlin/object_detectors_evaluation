from collections.abc import Iterator
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from object_detectors_evaluation.loggers import logger


class DetectionLatency(BaseModel):
    """Latency measurements for one detection inference call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    preprocess_ms: float = Field(ge=0, description="Input preprocessing latency in milliseconds.")
    inference_ms: float = Field(ge=0, description="Backend model inference latency in milliseconds.")
    postprocess_ms: float = Field(ge=0, description="Prediction postprocessing latency in milliseconds.")

    @property
    def total_ms(self) -> float:
        """Return total measured latency in milliseconds.

        :return: Sum of preprocess, inference, and postprocess latency.
        """
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


class DetectionPrediction(BaseModel):
    """Normalized object detection prediction for a single image."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    boxes: np.ndarray = Field(description="Predicted boxes in `XYXY` pixel coordinates on the original image.")
    scores: np.ndarray = Field(description="Confidence scores aligned with `boxes`.")
    labels: np.ndarray = Field(description="Engine-output integer class ids aligned with `boxes`.")
    labels_names: tuple[str, ...] | None = Field(
        default=None,
        description="Optional engine-output class names aligned with `labels`.",
    )
    image_id: int | str | None = Field(
        default=None,
        description="Optional dataset image id carried through by the evaluator.",
    )
    image_size: tuple[int, int] | None = Field(
        default=None,
        description="Optional original image size as `(height, width)`.",
    )
    class_space: str | None = Field(
        default=None,
        description="Optional class space produced by the model, such as `coco`.",
    )

    @field_validator("boxes", mode="before")
    @classmethod
    def validate_boxes_array(cls, value: Any) -> np.ndarray:
        """Convert prediction boxes to a float32 NumPy array.

        :param value: Raw boxes value.
        :return: Boxes array.
        """
        return np.asarray(value, dtype=np.float32)

    @field_validator("scores", mode="before")
    @classmethod
    def validate_scores_array(cls, value: Any) -> np.ndarray:
        """Convert prediction scores to a float32 NumPy array.

        :param value: Raw scores value.
        :return: Scores array.
        """
        return np.asarray(value, dtype=np.float32)

    @field_validator("labels", mode="before")
    @classmethod
    def validate_labels_array(cls, value: Any) -> np.ndarray:
        """Convert prediction labels to an int64 NumPy array.

        :param value: Raw labels value.
        :return: Labels array.
        """
        return np.asarray(value, dtype=np.int64)

    @model_validator(mode="after")
    def validate_prediction_shapes(self) -> Self:
        """Validate prediction array shapes after field conversion.

        :return: Validated prediction.
        """
        if self.boxes.ndim != 2 or self.boxes.shape[-1] != 4:
            msg = f"Prediction boxes must have shape `(N, 4)`, got `{self.boxes.shape}`"
            logger.error(msg)
            raise ValueError(msg)

        num_predictions = self.num_predictions
        if self.scores.ndim != 1 or self.scores.shape[0] != num_predictions:
            msg = (
                "Prediction scores must have shape `(N,)` matching boxes, "
                f"got `{self.scores.shape}` for `{num_predictions}` boxes"
            )
            logger.error(msg)
            raise ValueError(msg)

        if self.labels.ndim != 1 or self.labels.shape[0] != num_predictions:
            msg = (
                "Prediction labels must have shape `(N,)` matching boxes, "
                f"got `{self.labels.shape}` for `{num_predictions}` boxes"
            )
            logger.error(msg)
            raise ValueError(msg)

        if self.labels_names is not None and len(self.labels_names) != num_predictions:
            msg = (
                "Prediction label names must match number of boxes, "
                f"got `{len(self.labels_names)}` names for `{num_predictions}` boxes"
            )
            logger.error(msg)
            raise ValueError(msg)

        return self

    @property
    def num_predictions(self) -> int:
        """Return the number of predicted detections.

        :return: Number of predictions.
        """
        return int(self.boxes.shape[0])


class DetectionPredictionBatch(BaseModel):
    """Normalized object detection predictions for a batch of images."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    predictions: tuple[DetectionPrediction, ...] = Field(description="Per-image detection predictions.")
    latency: DetectionLatency | None = Field(
        default=None,
        description="Optional latency measurements for the inference call that produced this batch.",
    )

    def __len__(self) -> int:
        """Return the number of images in the prediction batch.

        :return: Batch size.
        """
        return len(self.predictions)

    def __iter__(self) -> Iterator[DetectionPrediction]:
        """Iterate over per-image predictions.

        :return: Iterator over predictions.
        """
        return iter(self.predictions)
