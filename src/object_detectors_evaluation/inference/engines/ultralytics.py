from collections.abc import Mapping
from typing import Any

import numpy as np
from ultralytics import YOLO

from object_detectors_evaluation.inference.base import BaseDetectionInferenceEngine, DetectionInputBatch, ImageId
from object_detectors_evaluation.inference.configs import DetectionInferenceConfig
from object_detectors_evaluation.inference.predictions import DetectionPrediction, DetectionPredictionBatch
from object_detectors_evaluation.models import DownloadedModel


class UltralyticsDetectionInferenceEngine(BaseDetectionInferenceEngine):
    """Detection inference engine for Ultralytics-compatible checkpoints."""

    engine_name = "ultralytics"
    supported_checkpoint_formats = ("pt",)

    def __init__(
        self,
        downloaded_model: DownloadedModel,
        config: DetectionInferenceConfig | None = None,
    ) -> None:
        super().__init__(downloaded_model=downloaded_model, config=config)

        self.predict_kwargs: dict[str, Any] = {
            "batch": self.config.batch_size,
            "conf": self.config.score_threshold,
            "verbose": False,
        }
        if self.config.device != "auto":
            self.predict_kwargs["device"] = self.config.device
        if self.config.max_detections is not None:
            self.predict_kwargs["max_det"] = self.config.max_detections
        if self.config.dtype == "float16":
            self.predict_kwargs["half"] = True

    def load_model(self) -> YOLO:
        """Load an Ultralytics model checkpoint.

        :return: Loaded Ultralytics model.
        """
        filepath = self.resolve_model_filepath(filesuffixes=(".pt",))
        return YOLO(filepath)

    @property
    def class_id_to_name(self) -> Mapping[int, str] | None:
        """Return Ultralytics label names keyed by label id.

        :return: Label-name mapping or ``None``.
        """
        names = getattr(self.model, "names", None)
        if names is None:
            return None

        if isinstance(names, dict):
            return {int(label): str(name) for label, name in names.items()}

        return {label: str(name) for label, name in enumerate(names)}

    def predict(self, preprocessed_batch: DetectionInputBatch) -> Any:
        """Run Ultralytics detection inference.

        :param preprocessed_batch: Common inference batch.
        :return: Raw Ultralytics results.
        """
        sources = list(preprocessed_batch.processed_inputs)
        output = self.model.predict(source=sources, **self.predict_kwargs)
        return output

    def postprocess(self, preprocessed_batch: DetectionInputBatch, raw_outputs: Any) -> DetectionPredictionBatch:
        """Convert Ultralytics results to normalized predictions.

        :param preprocessed_batch: Common inference batch.
        :param raw_outputs: Raw Ultralytics results.
        :return: Normalized detection predictions.
        """
        results = tuple(raw_outputs)
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
        result: Any,
        image_id: ImageId,
        image_size: tuple[int, int],
    ) -> DetectionPrediction:
        boxes_result = getattr(result, "boxes", None)
        if boxes_result is None or len(boxes_result) == 0:
            return DetectionPrediction(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                labels=np.empty((0,), dtype=np.int64),
                labels_names=(),
                image_id=image_id,
                image_size=image_size,
                class_space=self.class_space,
            )

        boxes = boxes_result.xyxy.cpu().numpy()
        scores = boxes_result.conf.cpu().numpy()
        labels = boxes_result.cls.cpu().numpy().astype(np.int64)
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
