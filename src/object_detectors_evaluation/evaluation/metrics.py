import json
from typing import Any

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

from object_detectors_evaluation.datasets.types import DetectionTarget
from object_detectors_evaluation.evaluation.configs import DetectionEvaluatorMetricsConfig
from object_detectors_evaluation.inference.predictions import DetectionPrediction


class DetectionMeanAveragePrecision:
    """TorchMetrics mAP wrapper for normalized detection predictions and targets.

    :param config: Mean average precision metric configuration.
    """

    def __init__(self, config: DetectionEvaluatorMetricsConfig) -> None:
        self.config = config
        self.metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            backend=config.backend,
            extended_summary=config.extended_summary,
            class_metrics=config.class_metrics,
        )

    def update(self, predictions: list[DetectionPrediction], targets: list[DetectionTarget]) -> None:
        """Update metric state with one prediction/target batch.

        :param predictions: Normalized predictions.
        :param targets: Dataset targets aligned with predictions.
        """
        self.metric.update(
            preds=[self._prediction_to_torchmetrics(prediction=prediction) for prediction in predictions],
            target=[self._target_to_torchmetrics(target=target) for target in targets],
        )

    def compute(self) -> dict[str, Any]:
        """Compute JSON-serializable metric values.

        :return: Metric output.
        """
        metrics = self.metric.compute()
        metrics_json = to_jsonable(value=metrics)
        return metrics_json

    def reset(self) -> None:
        """Reset accumulated metric state."""
        self.metric.reset()

    @staticmethod
    def _prediction_to_torchmetrics(prediction: DetectionPrediction) -> dict[str, torch.Tensor]:
        return {
            "boxes": torch.as_tensor(prediction.boxes, dtype=torch.float32),
            "scores": torch.as_tensor(prediction.scores, dtype=torch.float32),
            "labels": torch.as_tensor(prediction.labels, dtype=torch.int64),
        }

    @staticmethod
    def _target_to_torchmetrics(target: DetectionTarget) -> dict[str, torch.Tensor]:
        boxes = torch.as_tensor(target["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(target["labels"], dtype=torch.int64)
        metric_target = {
            "boxes": boxes,
            "labels": labels,
        }

        if "iscrowd" in target and len(target["iscrowd"]) == len(labels):
            metric_target["iscrowd"] = torch.as_tensor(target["iscrowd"], dtype=torch.int64)
        if "area" in target and len(target["area"]) == len(labels):
            metric_target["area"] = torch.as_tensor(target["area"], dtype=torch.float32)

        return metric_target


def to_jsonable(value: Any) -> Any:
    """Convert common metric values to JSON-serializable objects.

    :param value: Raw metric value.
    :return: JSON-serializable value.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {to_jsonable_key(key=key): to_jsonable(value=inner_value) for key, inner_value in value.items()}

    if isinstance(value, list | tuple):
        return [to_jsonable(value=inner_value) for inner_value in value]

    return value


def to_jsonable_key(key: Any) -> str:
    """Convert a metric dictionary key to a stable JSON object key.

    :param key: Raw dictionary key.
    :return: JSON object key.
    """
    key = to_jsonable(value=key)
    if isinstance(key, str):
        return key
    if isinstance(key, int | float | bool) or key is None:
        return str(key)
    return json.dumps(key, separators=(",", ":"))
