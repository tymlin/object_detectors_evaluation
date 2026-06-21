from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from object_detectors_evaluation.datasets.types import DatasetName, DatasetSplit


class DetectionEvaluationLatencySummary(BaseModel):
    """Aggregated latency measurements for one evaluated model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    preprocess_mean_ms: float = Field(ge=0, description="Mean preprocessing latency in milliseconds.")
    inference_mean_ms: float = Field(ge=0, description="Mean model inference latency in milliseconds.")
    postprocess_mean_ms: float = Field(ge=0, description="Mean postprocessing latency in milliseconds.")
    total_mean_ms: float = Field(ge=0, description="Mean total latency in milliseconds.")
    total_std_ms: float = Field(ge=0, description="Standard deviation of total batch latency in milliseconds.")
    total_min_ms: float = Field(ge=0, description="Minimum total batch latency in milliseconds.")
    total_p50_ms: float = Field(ge=0, description="50th percentile total batch latency in milliseconds.")
    total_p90_ms: float = Field(ge=0, description="90th percentile total batch latency in milliseconds.")
    total_p95_ms: float = Field(ge=0, description="95th percentile total batch latency in milliseconds.")
    total_p99_ms: float = Field(ge=0, description="99th percentile total batch latency in milliseconds.")
    total_max_ms: float = Field(ge=0, description="Maximum total batch latency in milliseconds.")
    throughput_images_per_second: float = Field(ge=0, description="Measured engine throughput in images per second.")
    num_images: int = Field(ge=0, description="Number of images included in latency measurements.")
    num_batches: int = Field(ge=0, description="Number of measured inference batches.")


class DetectionEvaluationModelResult(BaseModel):
    """Evaluation result for one model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str = Field(description="Evaluated model name.")
    engine_name: str = Field(description="Inference engine used for the model.")
    metrics: dict[str, Any] = Field(description="JSON-serializable metric output.")
    latency: DetectionEvaluationLatencySummary = Field(description="Aggregated model latency.")
    num_samples: int = Field(ge=0, description="Number of evaluated dataset samples.")
    run_dirpath: Path = Field(description="Model run output directory.")


class DetectionEvaluationRunResult(BaseModel):
    """Evaluation result for a full multi-model run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_name: str = Field(description="Resolved run name.")
    run_dirpath: Path = Field(description="Run output directory.")
    dataset_name: DatasetName = Field(description="Evaluated dataset name.")
    dataset_split: DatasetSplit = Field(description="Evaluated dataset split.")
    num_samples: int = Field(ge=0, description="Number of evaluated dataset samples.")
    models: tuple[DetectionEvaluationModelResult, ...] = Field(description="Per-model evaluation results.")
