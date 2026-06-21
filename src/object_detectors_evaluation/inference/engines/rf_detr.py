from collections.abc import Mapping
from typing import Any

import torch

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

    @property
    def _native_class_id_to_name(self) -> Mapping[int, str]:
        """Return RF-DETR checkpoint-native class names keyed by classifier id.

        :return: Native RF-DETR class mapping.
        """
        native_class_id_to_name = super().class_id_to_name
        if not native_class_id_to_name:
            msg = "RF-DETR model config must define `id2label` for canonical label conversion"
            logger.error(msg)
            raise ValueError(msg)

        return native_class_id_to_name

    @property
    def _native_class_id_to_class_id(self) -> Mapping[int, int]:
        """Map RF-DETR checkpoint-native ids to contiguous class ids.

        :return: Native-to-contiguous class-id mapping without unused ``N/A`` slots.
        """
        native_classes = sorted(
            (
                (native_class_id, class_name)
                for native_class_id, class_name in self._native_class_id_to_name.items()
                if class_name != "N/A"
            ),
            key=lambda item: item[0],
        )
        return {native_class_id: class_id for class_id, (native_class_id, _) in enumerate(native_classes)}

    @property
    def class_id_to_name(self) -> Mapping[int, str]:
        """Return canonical contiguous RF-DETR class names.

        :return: Canonical class-name mapping keyed by contiguous class id.
        """
        native_class_id_to_name = self._native_class_id_to_name
        native_class_id_to_class_id = self._native_class_id_to_class_id
        return {
            native_class_id_to_class_id[native_class_id]: class_name
            for native_class_id, class_name in native_class_id_to_name.items()
            if native_class_id in native_class_id_to_class_id
        }

    def _prediction_from_result(
        self,
        result: dict[str, Any],
        image_id: ImageId,
        image_size: tuple[int, int],
    ) -> DetectionPrediction:
        native_labels = result["labels"]
        native_label_values = native_labels.detach().cpu().tolist()
        native_class_id_to_class_id = self._native_class_id_to_class_id
        keep_mask = torch.tensor(
            [int(native_label) in native_class_id_to_class_id for native_label in native_label_values],
            dtype=torch.bool,
            device=native_labels.device,
        )
        labels = torch.tensor(
            [
                native_class_id_to_class_id[int(native_label)]
                for native_label in native_label_values
                if int(native_label) in native_class_id_to_class_id
            ],
            dtype=torch.int64,
            device=native_labels.device,
        )

        normalized_result = dict(result)
        normalized_result.update(
            boxes=result["boxes"][keep_mask],
            scores=result["scores"][keep_mask],
            labels=labels,
        )
        prediction = super()._prediction_from_result(
            result=normalized_result,
            image_id=image_id,
            image_size=image_size,
        )
        return prediction
