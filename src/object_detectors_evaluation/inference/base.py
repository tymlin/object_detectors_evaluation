from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL.Image import Image as PILImage

from object_detectors_evaluation.inference.configs import DetectionInferenceConfig
from object_detectors_evaluation.inference.predictions import DetectionPredictionBatch
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.models import DownloadedModel

ImageInput = PILImage | np.ndarray | str | Path
ImageId = int | str | None


class BaseDetectionInferenceEngine(ABC):
    """Base class for detection inference engines.

    :param downloaded_model: Local model artifacts and source metadata.
    :param config: Runtime-neutral inference config.
    """

    engine_name: ClassVar[str] = "base"
    supported_families: ClassVar[tuple[str, ...]] = ()
    supported_checkpoint_formats: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        downloaded_model: DownloadedModel,
        config: DetectionInferenceConfig | None = None,
    ) -> None:
        if downloaded_model.spec.task != "detection":
            msg = f"Model `{downloaded_model.spec.name}` has task `{downloaded_model.spec.task}`, expected `detection`"
            logger.error(msg)
            raise ValueError(msg)

        self.downloaded_model = downloaded_model
        self.config = config or DetectionInferenceConfig()
        logger.info(
            f"Initializing inference engine `{self.engine_name}` for model `{downloaded_model.spec.name}` "
            f"from path: '{downloaded_model.dirpath}'"
        )
        self.model = self.load_model()

    @property
    def class_space(self) -> str | None:
        """Return the model output class space.

        :return: Model class space.
        """
        return self.downloaded_model.spec.class_space

    @property
    def class_names(self) -> tuple[str, ...] | None:
        """Return model-native class names when the engine can expose them.

        :return: Model-native class names or ``None``.
        """
        return None

    @abstractmethod
    def load_model(self) -> object:
        """Load backend-specific model artifacts.

        :return: Loaded backend-specific model object.
        """

    def __call__(
        self,
        images: Sequence[ImageInput],
        image_ids: Sequence[ImageId] | None = None,
    ) -> DetectionPredictionBatch:
        """Run object detection for a batch of images.

        :param images: Images or image paths.
        :param image_ids: Optional image ids aligned with ``images``.
        :return: Normalized detection predictions.
        """
        return self.predict(images=images, image_ids=image_ids)

    @abstractmethod
    def predict(
        self,
        images: Sequence[ImageInput],
        image_ids: Sequence[ImageId] | None = None,
    ) -> DetectionPredictionBatch:
        """Run object detection for a batch of images.

        :param images: Images or image paths.
        :param image_ids: Optional image ids aligned with ``images``.
        :return: Normalized detection predictions.
        """

    def normalize_image_ids(
        self,
        images: Sequence[ImageInput],
        image_ids: Sequence[ImageId] | None,
    ) -> tuple[ImageId, ...]:
        """Normalize optional image ids to a tuple aligned with images.

        :param images: Input images.
        :param image_ids: Optional image ids.
        :return: Image ids aligned with ``images``.
        """
        if image_ids is None:
            return tuple(None for _ in images)

        if len(image_ids) != len(images):
            msg = f"Expected `{len(images)}` image ids, got `{len(image_ids)}`"
            logger.error(msg)
            raise ValueError(msg)

        return tuple(image_ids)

    def resolve_model_filepath(self, filesuffixes: tuple[str, ...] = ()) -> Path:
        """Resolve a single model artifact filepath from downloaded model files.

        :param filesuffixes: Optional allowed file suffixes.
        :return: Resolved model artifact filepath.
        """
        if self.downloaded_model.spec.resolved_filename is not None:
            filepath = self.downloaded_model.dirpath / self.downloaded_model.spec.resolved_filename
            if filepath.exists():
                return filepath

        filepaths = self.downloaded_model.filepaths
        if filesuffixes:
            filepaths = tuple(filepath for filepath in filepaths if filepath.suffix in filesuffixes)

        if len(filepaths) != 1:
            msg = (
                f"Could not resolve one model artifact for `{self.downloaded_model.spec.name}`, "
                f"found `{len(filepaths)}` candidates"
            )
            logger.error(msg)
            raise ValueError(msg)

        return filepaths[0]
