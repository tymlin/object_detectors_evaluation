from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from object_detectors_evaluation.datasets.types import DatasetName, DatasetSplit
from object_detectors_evaluation.reports.types import LatencyField


class DetectionReportConfig(BaseModel):
    """Resolved detection report configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_dirpath: Path = Field(description="Source evaluation run directory.")
    output_dirpath: Path = Field(description="Generated report directory.")
    score_threshold: float = Field(ge=0, le=1, description="Threshold for prediction-derived plots.")
    top_classes: int = Field(ge=1, description="Maximum classes shown in per-class plots.")
    model_names: tuple[str, ...] = Field(min_length=1, description="Models included in the report.")
    latency_field: LatencyField = Field(description="Latency field used by the accuracy-latency plot.")


class DetectionReportFigure(BaseModel):
    """Static report figure metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = Field(min_length=1, description="Display title.")
    filename: str = Field(min_length=1, description="Report-relative PNG filename.")
    description: str = Field(min_length=1, description="Figure description.")


class DetectionReportInteractivePlot(BaseModel):
    """Interactive report plot metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, description="Stable HTML plot identifier.")
    title: str = Field(min_length=1, description="Display title.")
    description: str = Field(min_length=1, description="Plot description.")


class DetectionReportData(BaseModel):
    """Serializable metadata for one generated detection report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generated_at: datetime = Field(description="Report generation timestamp.")
    run_name: str = Field(min_length=1, description="Evaluation run name.")
    dataset_name: DatasetName = Field(description="Evaluated dataset name.")
    dataset_split: DatasetSplit = Field(description="Evaluated dataset split.")
    num_samples: int = Field(ge=0, description="Number of evaluated samples.")
    num_models: int = Field(ge=1, description="Number of report models.")
    score_threshold: float = Field(ge=0, le=1, description="Threshold for prediction-derived plots.")
    latency_field: LatencyField = Field(description="Latency field used by the report.")
    warnings: tuple[str, ...] = Field(description="Comparison warnings.")
    leaderboard: tuple[dict[str, Any], ...] = Field(description="Normalized leaderboard records.")
    figures: tuple[DetectionReportFigure, ...] = Field(description="Static PNG figure metadata.")
    interactive_plots: tuple[DetectionReportInteractivePlot, ...] = Field(
        description="Interactive Plotly chart metadata."
    )
