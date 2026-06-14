from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

InferenceDevice = Literal["auto", "cpu", "cuda", "mps"]
InferenceDType = Literal["auto", "float32", "float16", "bfloat16"]


class DetectionInferenceConfig(BaseModel):
    """Runtime-neutral configuration shared by detection inference engines."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    device: InferenceDevice | str = Field(
        default="auto",
        description="Backend-specific device identifier. Engines decide how to interpret `auto`.",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Number of images passed to the backend in one inference call.",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Optional confidence threshold applied by the engine or postprocessor.",
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
