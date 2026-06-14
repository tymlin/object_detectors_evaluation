from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.models.types import ModelSourceType, ModelTask


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Model metadata used by registries, downloaders, inference engines, and evaluators.

    :param name: Stable model name used for local directories and result identifiers.
    :param family: Model family name, such as ``RT-DETR`` or ``YOLOv8``.
    :param source_type: Where model artifacts come from.
    :param task: Model task. Currently only detection models are tracked.
    :param repo_id: Hugging Face repository id for ``huggingface`` models.
    :param model_url: Model page URL for Hugging Face models or direct artifact URL for ``url`` models.
    :param filename: Expected local artifact filename for direct URL or local-file models.
    :param docs_url: Optional documentation URL.
    :param size: Optional model size variant, such as ``n``, ``s``, ``m``, or ``x``.
    :param training_track: Short training-track label from the source registry.
    :param class_space: Class label space produced by the model, such as ``coco``.
    :param training_dataset: Human-readable training dataset name when known.
    :param checkpoint_format: Artifact format, such as ``pt``, ``onnx``, or ``engine``.
    :param inference_engine: Default inference engine name for this model when known.
    :param note: Free-form source note.
    """

    name: str
    family: str
    source_type: ModelSourceType
    task: ModelTask = "detection"
    repo_id: str | None = None
    model_url: str | None = None
    filename: str | None = None
    docs_url: str | None = None
    size: str | None = None
    training_track: str | None = None
    class_space: str | None = None
    training_dataset: str | None = None
    checkpoint_format: str | None = None
    inference_engine: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.source_type == "huggingface" and self.repo_id is None:
            msg = f"Model `{self.name}` uses source type `huggingface` but has no repo_id"
            logger.error(msg)
            raise ValueError(msg)
        if self.source_type == "url" and self.model_url is None:
            msg = f"Model `{self.name}` uses source type `url` but has no model_url"
            logger.error(msg)
            raise ValueError(msg)
        if self.source_type == "local" and self.filename is None:
            msg = f"Model `{self.name}` uses source type `local` but has no filename"
            logger.error(msg)
            raise ValueError(msg)

    @property
    def resolved_filename(self) -> str | None:
        """Return the best local filename for the model artifact.

        :return: Explicit filename or filename derived from ``model_url`` when possible.
        """
        if self.filename is not None:
            return self.filename
        if self.model_url is None:
            return None
        return Path(self.model_url).name


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """Local model artifact produced by a downloader.

    :param spec: Source model spec.
    :param dirpath: Local model directory.
    :param filepaths: Local files associated with the downloaded model.
    """

    spec: ModelSpec
    dirpath: Path
    filepaths: tuple[Path, ...]


DownloadedModel = ModelArtifact
