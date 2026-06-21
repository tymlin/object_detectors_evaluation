from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from object_detectors_evaluation.inference.configs import DetectionInferenceConfig
from object_detectors_evaluation.inference.image_utils import (
    get_image_size,
    validate_image_input,
)
from object_detectors_evaluation.inference.predictions import DetectionLatency, DetectionPredictionBatch
from object_detectors_evaluation.inference.torch_utils import synchronize_torch_device
from object_detectors_evaluation.inference.types import ImageId, ImageInput, ProcessedInputs
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.models import ModelArtifact


@dataclass(slots=True)
class DetectionInputBatch:
    """Common detection input batch.

    :param processed_inputs: Validated image inputs passed to backend inference.
    :param image_ids: Optional image ids aligned with ``processed_inputs``.
    :param image_sizes: Original image sizes as ``(height, width)``.
    """

    processed_inputs: ProcessedInputs
    image_ids: list[ImageId]
    image_sizes: list[tuple[int, int]]


class BaseDetectionInferenceEngine(ABC):
    """Base class for detection inference engines.

    :param model_artifact: Local model artifact and source metadata.
    :param config: Runtime-neutral inference config.
    """

    engine_name = "base"
    supported_families: tuple[str, ...] = ()
    supported_checkpoint_formats: tuple[str, ...] = ()

    def __init__(
        self,
        model_artifact: ModelArtifact,
        config: DetectionInferenceConfig | None = None,
    ) -> None:
        if model_artifact.spec.task != "detection":
            msg = f"Model `{model_artifact.spec.name}` has task `{model_artifact.spec.task}`, expected `detection`"
            logger.error(msg)
            raise ValueError(msg)

        self.model_artifact = model_artifact
        self.config = config or DetectionInferenceConfig()
        logger.info(
            f"Initializing inference engine `{self.engine_name}` for model `{model_artifact.spec.name}` "
            f"from path: '{model_artifact.dirpath}' with config: {self.config.model_dump_json(indent=4)}"
        )
        self.model = self.load_model()
        logger.info(
            f"Initialized inference engine `{self.engine_name}` with model `{model_artifact.spec.name}` "
            f"on device `{self.device}` with dtype `{self.dtype}`"
        )

    @property
    def class_space(self) -> str | None:
        """Return the model output class space.

        :return: Model class space.
        """
        return self.model_artifact.spec.class_space

    @property
    def dtype(self) -> torch.dtype:
        """Return model parameter dtype.

        :return: Model dtype.
        """
        return next(self.model.parameters()).dtype

    @property
    def device(self) -> torch.device:
        """Return model parameter device.

        :return: Model device.
        """
        return next(self.model.parameters()).device

    @property
    def class_id_to_name(self) -> Mapping[int, str] | None:
        """Return engine-output class names keyed by class id.

        :return: Label-name mapping or ``None``.
        """
        return None

    @property
    def class_name_to_id(self) -> Mapping[str, int] | None:
        """Return engine-output class ids keyed by class name.

        :return: Class-name mapping or ``None``.
        """
        class_id_to_name = self.class_id_to_name
        if class_id_to_name is None:
            return None

        return {class_name: class_id for class_id, class_name in class_id_to_name.items()}

    @abstractmethod
    def load_model(self) -> Any:
        """Load backend-specific model artifacts.

        :return: Loaded backend-specific model object.
        """

    def __call__(
        self,
        images: Sequence[ImageInput],
        image_ids: Sequence[ImageId] | None = None,
    ) -> DetectionPredictionBatch:
        """Run object detection for a batch of images.

        :param images: NumPy arrays or torch tensors.
        :param image_ids: Optional image ids aligned with ``images``.
        :return: Normalized detection predictions.
        """
        preprocess_start_time = perf_counter()
        preprocessed_batch = self.preprocess(images=images, image_ids=image_ids)
        preprocess_ms = (perf_counter() - preprocess_start_time) * 1000.0

        self.synchronize()
        inference_start_time = perf_counter()
        raw_outputs = self.predict(preprocessed_batch=preprocessed_batch)
        self.synchronize()
        inference_ms = (perf_counter() - inference_start_time) * 1000.0

        postprocess_start_time = perf_counter()
        prediction_batch = self.postprocess(preprocessed_batch=preprocessed_batch, raw_outputs=raw_outputs)
        postprocess_ms = (perf_counter() - postprocess_start_time) * 1000.0

        latency = DetectionLatency(
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
        )
        return prediction_batch.model_copy(update={"latency": latency})

    def preprocess(
        self,
        images: Sequence[ImageInput],
        image_ids: Sequence[ImageId] | None = None,
    ) -> DetectionInputBatch:
        """Validate images and create a common inference batch.

        :param images: NumPy arrays or torch tensors.
        :param image_ids: Optional image ids aligned with ``images``.
        :return: Common inference batch.
        """
        if isinstance(images, (np.ndarray, torch.Tensor)):
            msg = "Inference images must be a sequence of per-image inputs, for example `[image]`"
            logger.error(msg)
            raise TypeError(msg)

        normalized_image_ids = self.normalize_image_ids(images=images, image_ids=image_ids)
        processed_inputs = [validate_image_input(image=image) for image in images]
        image_sizes = [get_image_size(image=image) for image in processed_inputs]
        return DetectionInputBatch(
            processed_inputs=processed_inputs,
            image_ids=normalized_image_ids,
            image_sizes=image_sizes,
        )

    @abstractmethod
    def predict(self, preprocessed_batch: DetectionInputBatch) -> Any:
        """Run backend inference on a preprocessed batch.

        :param preprocessed_batch: Common inference batch.
        :return: Backend-specific raw model outputs.
        """

    @abstractmethod
    def postprocess(self, preprocessed_batch: DetectionInputBatch, raw_outputs: Any) -> DetectionPredictionBatch:
        """Convert backend raw outputs to normalized detection predictions.

        :param preprocessed_batch: Common inference batch.
        :param raw_outputs: Backend-specific raw model outputs.
        :return: Normalized detection predictions.
        """

    def synchronize(self) -> None:
        """Synchronize backend runtime before/after inference timing.

        Engines with asynchronous device execution should override this method.
        """
        synchronize_torch_device(device=self.device)

    def normalize_image_ids(
        self,
        images: Sequence[ImageInput],
        image_ids: Sequence[ImageId] | None,
    ) -> list[ImageId]:
        """Normalize optional image ids to a list aligned with images.

        :param images: Input images.
        :param image_ids: Optional image ids.
        :return: Image ids aligned with ``images``.
        """
        if image_ids is None:
            return [None for _ in images]

        if len(image_ids) != len(images):
            msg = f"Expected `{len(images)}` image ids, got `{len(image_ids)}`"
            logger.error(msg)
            raise ValueError(msg)

        return list(image_ids)

    def get_label_name(self, label: int) -> str:
        """Return an engine-output label name.

        :param label: Engine-output integer label id.
        :return: Label name when known, otherwise the label id as a string.
        """
        class_id_to_name = self.class_id_to_name
        if class_id_to_name is None:
            return str(label)

        return str(class_id_to_name.get(int(label), label))

    def resolve_model_filepath(self, filesuffixes: tuple[str, ...] = ()) -> Path:
        """Resolve a single model artifact filepath from downloaded model files.

        :param filesuffixes: Optional allowed file suffixes.
        :return: Resolved model artifact filepath.
        """
        if self.model_artifact.spec.resolved_filename is not None:
            filepath = self.model_artifact.dirpath / self.model_artifact.spec.resolved_filename
            if filepath.exists():
                return filepath

        filepaths = self.model_artifact.filepaths
        if filesuffixes:
            filepaths = tuple(filepath for filepath in filepaths if filepath.suffix in filesuffixes)

        if len(filepaths) != 1:
            msg = (
                f"Could not resolve one model artifact for `{self.model_artifact.spec.name}`, "
                f"found `{len(filepaths)}` candidates"
            )
            logger.error(msg)
            raise ValueError(msg)

        return filepaths[0]
