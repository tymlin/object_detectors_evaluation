from collections.abc import Mapping
from functools import cached_property
from typing import Any

import numpy as np

from object_detectors_evaluation.inference.engines.transformers import TransformersDetectionInferenceEngine
from object_detectors_evaluation.inference.predictions import DetectionPrediction
from object_detectors_evaluation.inference.types import ImageId
from object_detectors_evaluation.loggers import logger


class RFDetrDetectionInferenceEngine(TransformersDetectionInferenceEngine):
    """Transformers inference engine with canonical RF-DETR COCO labels.

    Official RF-DETR COCO checkpoints use sparse COCO category ids with unused
    ``N/A`` classifier slots. This adapter removes those slots and compacts the
    remaining model-native ids into the canonical contiguous COCO label range.
    """

    engine_name = "rf_detr"

    @cached_property
    def native_class_id_to_name(self) -> Mapping[int, str]:
        """Return RF-DETR checkpoint-native class names keyed by classifier id.

        :return: Native RF-DETR class mapping.
        """
        id2label = getattr(getattr(self.model, "config", None), "id2label", None)
        if not id2label:
            msg = "RF-DETR model config must define `id2label` for canonical label conversion"
            logger.error(msg)
            raise ValueError(msg)

        return {int(class_id): str(class_name) for class_id, class_name in id2label.items()}

    @cached_property
    def native_class_id_to_class_id(self) -> Mapping[int, int]:
        """Map RF-DETR checkpoint-native ids to contiguous class ids.

        :return: Native-to-contiguous class-id mapping without unused ``N/A`` slots.
        """
        native_classes = sorted(
            (
                (native_class_id, class_name)
                for native_class_id, class_name in self.native_class_id_to_name.items()
                if class_name != "N/A"
            ),
            key=lambda item: item[0],
        )
        return {native_class_id: class_id for class_id, (native_class_id, _) in enumerate(native_classes)}

    @cached_property
    def class_id_to_name(self) -> Mapping[int, str]:
        """Return canonical contiguous RF-DETR class names.

        :return: Canonical class-name mapping keyed by contiguous class id.
        """
        return {
            self.native_class_id_to_class_id[native_class_id]: class_name
            for native_class_id, class_name in self.native_class_id_to_name.items()
            if native_class_id in self.native_class_id_to_class_id
        }

    def _prediction_from_result(
        self,
        result: dict[str, Any],
        image_id: ImageId,
        image_size: tuple[int, int],
    ) -> DetectionPrediction:
        boxes = result["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = result["scores"].detach().cpu().numpy().astype(np.float32)
        native_labels = result["labels"].detach().cpu().numpy().astype(np.int64)

        keep_mask = np.asarray(
            [int(native_label) in self.native_class_id_to_class_id for native_label in native_labels],
            dtype=np.bool_,
        )
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        labels = np.asarray(
            [self.native_class_id_to_class_id[int(native_label)] for native_label in native_labels[keep_mask]],
            dtype=np.int64,
        )

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
