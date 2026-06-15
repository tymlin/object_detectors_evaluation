from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DetectionEvaluationLatencySummary(BaseModel):
    """Aggregated latency measurements for one evaluated model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    preprocess_mean_ms: float = Field(ge=0, description="Mean preprocessing latency in milliseconds.")
    inference_mean_ms: float = Field(ge=0, description="Mean model inference latency in milliseconds.")
    postprocess_mean_ms: float = Field(ge=0, description="Mean postprocessing latency in milliseconds.")
    total_mean_ms: float = Field(ge=0, description="Mean total latency in milliseconds.")
    num_batches: int = Field(ge=0, description="Number of measured inference batches.")


class DetectionEvaluationModelResult(BaseModel):
    """Evaluation result for one model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str = Field(description="Evaluated model name.")
    engine_name: str = Field(description="Inference engine used for the model.")
    metrics: dict[str, Any] = Field(description="JSON-serializable metric output.")
    latency: DetectionEvaluationLatencySummary = Field(description="Aggregated model latency.")
    num_samples: int = Field(ge=0, description="Number of evaluated dataset samples.")
    run_dirpath: str = Field(description="Model run output directory.")


class DetectionEvaluationRunResult(BaseModel):
    """Evaluation result for a full multi-model run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_name: str = Field(description="Resolved run name.")
    run_dirpath: str = Field(description="Run output directory.")
    dataset_name: str = Field(description="Evaluated dataset name.")
    dataset_split: str = Field(description="Evaluated dataset split.")
    num_samples: int = Field(ge=0, description="Number of evaluated dataset samples.")
    models: tuple[DetectionEvaluationModelResult, ...] = Field(description="Per-model evaluation results.")
