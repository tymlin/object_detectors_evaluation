from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from object_detectors_evaluation.datasets.configs import DetectionDatasetConfig
from object_detectors_evaluation.datasets.types import DatasetName
from object_detectors_evaluation.evaluation.types import MeanAveragePrecisionBackend
from object_detectors_evaluation.inference.configs import DetectionInferenceConfig
from object_detectors_evaluation.utils.files import load_yaml


class DetectionEvaluatorDatasetConfig(BaseModel):
    """Dataset configuration used by a detection evaluation run.

    :param name: Dataset implementation name.
    :param auto_download: Whether evaluator may download the dataset before loading it.
    :param config: Dataset construction config.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: DatasetName = Field(description="Dataset implementation name.")
    auto_download: bool = Field(
        default=False,
        description="Whether evaluator may download the dataset before loading it.",
    )
    config: DetectionDatasetConfig = Field(description="Dataset construction config.")


class DetectionEvaluatorModelConfig(BaseModel):
    """Model configuration used by a detection evaluation run.

    :param name: Model registry name.
    :param engine_name: Optional inference engine override.
    :param auto_download: Whether evaluator may download the model before loading it.
    :param config: Inference engine config.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Model registry name.")
    engine_name: str | None = Field(
        default=None,
        description="Optional inference engine override.",
    )
    auto_download: bool = Field(
        default=True,
        description="Whether evaluator may download the model before loading it.",
    )
    config: DetectionInferenceConfig = Field(
        default_factory=DetectionInferenceConfig,
        description="Inference engine config.",
    )


class DetectionEvaluatorMetricsConfig(BaseModel):
    """Mean average precision metric configuration.

    :param backend: TorchMetrics mAP backend.
    :param extended_summary: Whether to return extended metric summary.
    :param class_metrics: Whether to return per-class metrics.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: MeanAveragePrecisionBackend = Field(
        default="pycocotools",
        description="TorchMetrics mAP backend.",
    )
    extended_summary: bool = Field(
        default=True,
        description="Whether to return extended metric summary.",
    )
    class_metrics: bool = Field(
        default=True,
        description="Whether to return per-class metrics.",
    )


class DetectionEvaluatorOutputsConfig(BaseModel):
    """Output artifact configuration for a detection evaluation run.

    :param save_predictions: Whether to save per-image predictions as JSON lines.
    :param save_plots: Whether to save prediction and target-prediction plots.
    :param num_plot_samples: Optional maximum number of evaluated samples to plot. ``None`` means all samples.
    :param plot_score_threshold: Score threshold used only for plotting. ``None`` uses model score threshold.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    save_predictions: bool = Field(
        default=False,
        description="Whether to save per-image predictions as JSON lines.",
    )
    save_plots: bool = Field(
        default=True,
        description="Whether to save prediction and target-prediction plots.",
    )
    num_plot_samples: int | None = Field(
        default=10,
        ge=0,
        description="Optional maximum number of evaluated samples to plot. `None` means all samples.",
    )
    plot_score_threshold: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Score threshold used only for plotting. `None` uses model score threshold.",
    )


class DetectionEvaluatorRuntimeConfig(BaseModel):
    """Runtime configuration for the evaluator loop.

    :param batch_size: Number of dataset samples to evaluate per engine call.
    :param num_samples: Optional number of dataset samples to evaluate. ``None`` means all samples.
    :param warmup_iterations: Number of warmup engine calls before measuring a model.
    :param map_progress_update_interval: Optional number of batches between running mAP progress updates.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_size: int = Field(
        default=1,
        ge=1,
        description="Number of dataset samples to evaluate per engine call.",
    )
    num_samples: int | None = Field(
        default=None,
        ge=1,
        description="Optional number of dataset samples to evaluate. `None` means all samples.",
    )
    warmup_iterations: int = Field(
        default=3,
        ge=0,
        description="Number of warmup engine calls before measuring a model.",
    )
    map_progress_update_interval: int | None = Field(
        default=100,
        ge=1,
        description="Optional number of batches between running mAP progress updates. `None` disables them.",
    )
    metrics: DetectionEvaluatorMetricsConfig = Field(
        default_factory=DetectionEvaluatorMetricsConfig,
        description="Mean average precision metric configuration.",
    )


class DetectionEvaluatorConfig(BaseModel):
    """Top-level detection evaluator config.

    :param run_name_postfix: Optional path-safe postfix appended to the timestamped run name.
    :param dataset: Dataset configuration.
    :param models: Model configurations to evaluate on the dataset.
    :param evaluation: Evaluator runtime and metric configuration.
    :param outputs: Output artifact configuration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_name_postfix: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
        description="Optional path-safe postfix appended to the timestamped run name.",
    )
    dataset: DetectionEvaluatorDatasetConfig = Field(description="Dataset configuration.")
    models: tuple[DetectionEvaluatorModelConfig, ...] = Field(
        min_length=1,
        description="Model configurations to evaluate on the dataset.",
    )
    evaluation: DetectionEvaluatorRuntimeConfig = Field(
        default_factory=DetectionEvaluatorRuntimeConfig,
        description="Evaluator runtime and metric configuration.",
    )
    outputs: DetectionEvaluatorOutputsConfig = Field(
        default_factory=DetectionEvaluatorOutputsConfig,
        description="Output artifact configuration.",
    )

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> "DetectionEvaluatorConfig":
        """Load evaluator config from a YAML file.

        :param filepath: YAML config filepath.
        :return: Parsed evaluator config.
        """
        data = load_yaml(filepath=filepath)
        # ``defaults`` is a YAML-only section used for anchors and aliases, not part of the runtime schema.
        data.pop("defaults", None)
        return cls.model_validate(data)
