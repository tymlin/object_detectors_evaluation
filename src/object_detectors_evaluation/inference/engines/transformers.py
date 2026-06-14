from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as PILImage
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from object_detectors_evaluation.inference.base import BaseDetectionInferenceEngine, ImageId, ImageInput
from object_detectors_evaluation.inference.predictions import DetectionPrediction, DetectionPredictionBatch
from object_detectors_evaluation.loggers import logger


class TransformersDetectionInferenceEngine(BaseDetectionInferenceEngine):
    """Detection inference engine for Hugging Face Transformers object detectors."""

    engine_name = "transformers"
    supported_checkpoint_formats = ("safetensors",)

    def load_model(self) -> object:
        """Load a Hugging Face Transformers object detection model.

        :return: Loaded Transformers model.
        """
        self.processor = AutoImageProcessor.from_pretrained(self.downloaded_model.dirpath)
        model = AutoModelForObjectDetection.from_pretrained(self.downloaded_model.dirpath)
        model = model.to(self._resolve_torch_device())

        torch_dtype = self._resolve_torch_dtype()
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)

        model.eval()
        return model

    @property
    def class_names(self) -> tuple[str, ...] | None:
        """Return model-native class names when available.

        :return: Model-native class names or ``None``.
        """
        id2label = getattr(getattr(self.model, "config", None), "id2label", None)
        if id2label is None:
            return None
        return tuple(str(id2label[idx]) for idx in sorted(id2label))

    def predict(
        self,
        images: Sequence[ImageInput],
        image_ids: Sequence[ImageId] | None = None,
    ) -> DetectionPredictionBatch:
        """Run Transformers object detection inference.

        :param images: Images or image paths.
        :param image_ids: Optional image ids aligned with ``images``.
        :return: Normalized detection predictions.
        """
        normalized_image_ids = self.normalize_image_ids(images=images, image_ids=image_ids)
        loaded_images = tuple(self._load_image(image=image) for image in images)
        image_sizes = tuple(self._image_size(image=image) for image in loaded_images)
        inputs = self.processor(images=list(loaded_images), return_tensors="pt")
        inputs = {name: value.to(self.model.device) for name, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        threshold = 0.0 if self.config.score_threshold is None else self.config.score_threshold
        target_sizes = torch.tensor(image_sizes, device=self.model.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=threshold,
            target_sizes=target_sizes,
        )
        predictions = tuple(
            self._prediction_from_result(result=result, image_id=image_id, image_size=image_size)
            for result, image_id, image_size in zip(results, normalized_image_ids, image_sizes, strict=True)
        )
        return DetectionPredictionBatch(predictions=predictions)

    def _prediction_from_result(
        self,
        result: dict[str, object],
        image_id: ImageId,
        image_size: tuple[int, int],
    ) -> DetectionPrediction:
        boxes = result["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = result["scores"].detach().cpu().numpy().astype(np.float32)
        labels = result["labels"].detach().cpu().numpy().astype(np.int64)

        if self.config.max_detections is not None:
            boxes = boxes[: self.config.max_detections]
            scores = scores[: self.config.max_detections]
            labels = labels[: self.config.max_detections]

        labels_names = tuple(self._label_name(label=label) for label in labels)
        return DetectionPrediction(
            boxes=boxes,
            scores=scores,
            labels=labels,
            labels_names=labels_names,
            image_id=image_id,
            image_size=image_size,
            class_space=self.class_space,
        )

    def _resolve_torch_device(self) -> torch.device:
        if self.config.device != "auto":
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_torch_dtype(self) -> torch.dtype | None:
        if self.config.dtype in (None, "auto"):
            return None

        dtype_by_name = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.config.dtype not in dtype_by_name:
            msg = f"Transformers engine does not support dtype `{self.config.dtype}`"
            logger.error(msg)
            raise ValueError(msg)
        return dtype_by_name[self.config.dtype]

    def _label_name(self, label: int) -> str:
        id2label = getattr(getattr(self.model, "config", None), "id2label", None)
        if id2label is None:
            return str(label)
        return str(id2label.get(int(label), label))

    @staticmethod
    def _load_image(image: ImageInput) -> PILImage | np.ndarray:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        return image

    @staticmethod
    def _image_size(image: PILImage | np.ndarray) -> tuple[int, int]:
        if isinstance(image, PILImage):
            return image.height, image.width
        height, width = image.shape[:2]
        return int(height), int(width)
