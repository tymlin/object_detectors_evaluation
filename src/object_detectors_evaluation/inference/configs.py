from pydantic import BaseModel, ConfigDict, Field

from object_detectors_evaluation.inference.types import InferenceDevice, InferenceDType


class DetectionInferenceConfig(BaseModel):
    """Runtime-neutral configuration shared by detection inference engines."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    device: InferenceDevice | str = Field(
        default="auto",
        description="Backend-specific device identifier. Engines decide how to interpret `auto`.",
    )
    score_threshold: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Confidence threshold applied by the engine or postprocessor.",
    )
    max_detections: int | None = Field(
        default=None,
        ge=1,
        description="Optional maximum number of detections returned per image.",
    )
    dtype: InferenceDType | str | None = Field(
        default=None,
        description="Optional backend-specific precision hint. Engines decide how to interpret it.",
    )
