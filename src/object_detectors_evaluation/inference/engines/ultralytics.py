from collections.abc import Sequence

import numpy as np
from ultralytics import YOLO

from object_detectors_evaluation.inference.base import BaseDetectionInferenceEngine, ImageId, ImageInput
from object_detectors_evaluation.inference.predictions import DetectionPrediction, DetectionPredictionBatch


class UltralyticsDetectionInferenceEngine(BaseDetectionInferenceEngine):
    """Detection inference engine for Ultralytics-compatible checkpoints."""

    engine_name = "ultralytics"
    supported_checkpoint_formats = ("pt",)

    def load_model(self) -> object:
        """Load an Ultralytics model checkpoint.

        :return: Loaded Ultralytics model.
        """
        filepath = self.resolve_model_filepath(filesuffixes=(".pt",))
        return YOLO(filepath)

    @property
    def class_names(self) -> tuple[str, ...] | None:
        """Return model-native class names when available.

        :return: Model-native class names or ``None``.
        """
        names = getattr(self.model, "names", None)
        if names is None:
            return None

        if isinstance(names, dict):
            return tuple(str(names[idx]) for idx in sorted(names))

        return tuple(str(name) for name in names)

    def predict(
        self,
        images: Sequence[ImageInput],
        image_ids: Sequence[ImageId] | None = None,
    ) -> DetectionPredictionBatch:
        """Run Ultralytics detection inference.

        :param images: Images or image paths.
        :param image_ids: Optional image ids aligned with ``images``.
        :return: Normalized detection predictions.
        """
        normalized_image_ids = self.normalize_image_ids(images=images, image_ids=image_ids)
        predict_kwargs = {
            "batch": self.config.batch_size,
            "verbose": False,
        }
        if self.config.device != "auto":
            predict_kwargs["device"] = self.config.device
        if self.config.score_threshold is not None:
            predict_kwargs["conf"] = self.config.score_threshold
        if self.config.max_detections is not None:
            predict_kwargs["max_det"] = self.config.max_detections
        if self.config.dtype == "float16":
            predict_kwargs["half"] = True

        sources = tuple(str(image) if hasattr(image, "__fspath__") else image for image in images)
        results = self.model.predict(source=list(sources), **predict_kwargs)
        predictions = tuple(
            self._prediction_from_result(result=result, image_id=image_id)
            for result, image_id in zip(results, normalized_image_ids, strict=True)
        )
        return DetectionPredictionBatch(predictions=predictions)

    def _prediction_from_result(self, result: object, image_id: ImageId) -> DetectionPrediction:
        boxes_result = getattr(result, "boxes", None)
        if boxes_result is None or len(boxes_result) == 0:
            return DetectionPrediction(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                labels=np.empty((0,), dtype=np.int64),
                labels_names=(),
                image_id=image_id,
                image_size=self._image_size_from_result(result=result),
                class_space=self.class_space,
            )

        boxes = boxes_result.xyxy.cpu().numpy()
        scores = boxes_result.conf.cpu().numpy()
        labels = boxes_result.cls.cpu().numpy().astype(np.int64)
        labels_names = tuple(self._label_name(label=label) for label in labels)
        return DetectionPrediction(
            boxes=boxes,
            scores=scores,
            labels=labels,
            labels_names=labels_names,
            image_id=image_id,
            image_size=self._image_size_from_result(result=result),
            class_space=self.class_space,
        )

    def _label_name(self, label: int) -> str:
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            return str(names.get(int(label), label))
        if names is not None and 0 <= int(label) < len(names):
            return str(names[int(label)])
        return str(label)

    @staticmethod
    def _image_size_from_result(result: object) -> tuple[int, int] | None:
        image_size = getattr(result, "orig_shape", None)
        if image_size is None:
            return None
        height, width = image_size
        return int(height), int(width)
