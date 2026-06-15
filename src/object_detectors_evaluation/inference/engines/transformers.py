from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from object_detectors_evaluation.inference.base import BaseDetectionInferenceEngine, DetectionInputBatch
from object_detectors_evaluation.inference.configs import DetectionInferenceConfig
from object_detectors_evaluation.inference.predictions import DetectionPrediction, DetectionPredictionBatch
from object_detectors_evaluation.inference.torch_utils import (
    resolve_torch_device,
    resolve_torch_dtype,
)
from object_detectors_evaluation.inference.types import ImageId, ImageInput
from object_detectors_evaluation.models import ModelArtifact


class TransformersDetectionInferenceEngine(BaseDetectionInferenceEngine):
    """Detection inference engine for Hugging Face Transformers object detectors."""

    engine_name = "transformers"
    supported_checkpoint_formats = ("safetensors",)

    def __init__(
        self,
        model_artifact: ModelArtifact,
        config: DetectionInferenceConfig | None = None,
    ) -> None:
        engine_config = config or DetectionInferenceConfig()
        self.resolved_device = resolve_torch_device(device=engine_config.device)
        self.resolved_dtype = resolve_torch_dtype(dtype=engine_config.dtype)
        self.max_detections = engine_config.max_detections

        super().__init__(model_artifact=model_artifact, config=engine_config)

    def load_model(self) -> Any:
        """Load a Hugging Face Transformers object detection model.

        :return: Loaded Transformers model.
        """
        self.processor = AutoImageProcessor.from_pretrained(self.model_artifact.dirpath)
        model = AutoModelForObjectDetection.from_pretrained(self.model_artifact.dirpath)
        model = model.to(self.resolved_device)

        if self.resolved_dtype is not None:
            model = model.to(dtype=self.resolved_dtype)

        model.eval()
        return model

    @property
    def class_id_to_name(self) -> Mapping[int, str] | None:
        """Return Transformers label names keyed by label id.

        :return: Label-name mapping or ``None``.
        """
        id2label = getattr(getattr(self.model, "config", None), "id2label", None)
        if id2label is None:
            return None
        return {int(label): str(name) for label, name in id2label.items()}

    def preprocess(
        self,
        images: Sequence[ImageInput],
        image_ids: Sequence[ImageId] | None = None,
    ) -> DetectionInputBatch:
        """Preprocess images for Transformers inference.

        :param images: NumPy arrays or torch tensors.
        :param image_ids: Optional image ids aligned with ``images``.
        :return: Transformers input batch.
        """
        input_batch = super().preprocess(images=images, image_ids=image_ids)
        inputs = self.processor(images=input_batch.processed_inputs, return_tensors="pt")
        inputs = {name: value.to(self.resolved_device) for name, value in inputs.items()}
        return DetectionInputBatch(
            processed_inputs=inputs,
            image_ids=input_batch.image_ids,
            image_sizes=input_batch.image_sizes,
        )

    def predict(self, preprocessed_batch: DetectionInputBatch) -> Any:
        """Run Transformers object detection inference.

        :param preprocessed_batch: Common inference batch.
        :return: Raw Transformers outputs.
        """
        inputs = preprocessed_batch.processed_inputs

        with torch.inference_mode():
            output = self.model(**inputs)

        return output

    def postprocess(self, preprocessed_batch: DetectionInputBatch, raw_outputs: Any) -> DetectionPredictionBatch:
        """Convert Transformers outputs to normalized predictions.

        :param preprocessed_batch: Common inference batch.
        :param raw_outputs: Raw Transformers outputs.
        :return: Normalized detection predictions.
        """
        target_sizes = torch.tensor(preprocessed_batch.image_sizes, device=self.resolved_device)
        results = self.processor.post_process_object_detection(
            outputs=raw_outputs,
            threshold=self.config.score_threshold,
            target_sizes=target_sizes,
        )
        predictions = tuple(
            self._prediction_from_result(result=result, image_id=image_id, image_size=image_size)
            for result, image_id, image_size in zip(
                results,
                preprocessed_batch.image_ids,
                preprocessed_batch.image_sizes,
                strict=True,
            )
        )
        return DetectionPredictionBatch(predictions=predictions)

    def _prediction_from_result(
        self,
        result: dict[str, Any],
        image_id: ImageId,
        image_size: tuple[int, int],
    ) -> DetectionPrediction:
        boxes = result["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = result["scores"].detach().cpu().numpy().astype(np.float32)
        labels = result["labels"].detach().cpu().numpy().astype(np.int64)

        if self.max_detections is not None:
            boxes = boxes[: self.max_detections]
            scores = scores[: self.max_detections]
            labels = labels[: self.max_detections]

        labels_names = tuple(self.get_label_name(label=label) for label in labels)
        return DetectionPrediction(
            boxes=boxes,
            scores=scores,
            labels=labels,
            labels_names=labels_names,
            image_id=image_id,
            image_size=image_size,
            class_space=self.class_space,
        )
