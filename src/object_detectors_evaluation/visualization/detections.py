from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from distinctipy import distinctipy
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image

from object_detectors_evaluation.datasets.types import DetectionTarget
from object_detectors_evaluation.inference.predictions import DetectionPrediction
from object_detectors_evaluation.inference.types import ImageInput
from object_detectors_evaluation.visualization.types import Color, ColorKey, ColorMap


def plot_prediction(
    image: ImageInput | Image.Image,
    prediction: DetectionPrediction,
    score_threshold: float | None = None,
    colors: ColorMap | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 8.0),
) -> Figure:
    """Plot model predictions on one image.

    :param image: RGB image as ``HWC`` NumPy array, ``CHW`` torch tensor, or PIL image.
    :param prediction: Normalized prediction for the image.
    :param score_threshold: Optional minimum score to draw.
    :param colors: Optional colors keyed by class id or class name.
    :param title: Optional axis title.
    :param figsize: Matplotlib figure size.
    :return: Matplotlib figure.
    """
    image_array = _image_to_numpy(image=image)
    fig, axis = plt.subplots(1, 1, figsize=figsize)
    axis.imshow(image_array)
    axis.axis("off")
    axis.set_title(title or _prediction_title(prediction=prediction))

    items = _prediction_items(prediction=prediction, score_threshold=score_threshold)
    colors_by_key = _resolve_colors(items=items, colors=colors)
    _draw_items(axis=axis, items=items, colors=colors_by_key, linestyle="-")
    fig.tight_layout()
    return fig


def plot_target_prediction(
    image: ImageInput | Image.Image,
    target: DetectionTarget,
    prediction: DetectionPrediction,
    score_threshold: float | None = None,
    colors: ColorMap | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 8.0),
) -> Figure:
    """Plot dataset target boxes and model predictions on one image.

    :param image: RGB image as ``HWC`` NumPy array, ``CHW`` torch tensor, or PIL image.
    :param target: Dataset detection target dictionary.
    :param prediction: Normalized prediction for the image.
    :param score_threshold: Optional minimum prediction score to draw.
    :param colors: Optional colors keyed by class id or class name.
    :param title: Optional axis title.
    :param figsize: Matplotlib figure size.
    :return: Matplotlib figure.
    """
    image_array = _image_to_numpy(image=image)
    fig, axis = plt.subplots(1, 1, figsize=figsize)
    axis.imshow(image_array)
    axis.axis("off")
    axis.set_title(title or _target_prediction_title(target=target, prediction=prediction))

    target_items = _target_items(target=target)
    prediction_items = _prediction_items(prediction=prediction, score_threshold=score_threshold)
    colors_by_key = _resolve_colors(items=target_items + prediction_items, colors=colors)

    _draw_items(axis=axis, items=target_items, colors=colors_by_key, linestyle="-", prefix="target")
    _draw_items(axis=axis, items=prediction_items, colors=colors_by_key, linestyle="--", prefix="pred")
    fig.tight_layout()
    return fig


def _image_to_numpy(image: ImageInput | Image.Image) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3 and image.shape[0] in {1, 3}:
            image = image.permute(1, 2, 0)
        return image.numpy()

    return np.asarray(image)


def _array_to_numpy(value: Any, dtype: np.dtype | type | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _prediction_items(
    prediction: DetectionPrediction,
    score_threshold: float | None,
) -> list[dict[str, Any]]:
    boxes = prediction.boxes
    scores = prediction.scores
    labels = prediction.labels
    labels_names = prediction.labels_names
    items = []

    for index, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        score_float = float(score)
        if score_threshold is not None and score_float < score_threshold:
            continue

        label_int = int(label)
        label_name = labels_names[index] if labels_names is not None else str(label_int)
        color_key = label_name if labels_names is not None else label_int
        items.append(
            {
                "box": box,
                "label": f"{label_name} {score_float:.2f}",
                "label_id": label_int,
                "color_key": color_key,
            }
        )

    return items


def _target_items(target: DetectionTarget) -> list[dict[str, Any]]:
    boxes = _array_to_numpy(value=target["boxes"], dtype=np.float32)
    labels = _array_to_numpy(value=target["labels"], dtype=np.int64)
    labels_names = target.get("labels_names")
    items = []

    for index, (box, label) in enumerate(zip(boxes, labels)):
        label_int = int(label)
        label_name = labels_names[index] if labels_names is not None else str(label_int)
        color_key = label_name if labels_names is not None else label_int
        items.append(
            {
                "box": box,
                "label": str(label_name),
                "label_id": label_int,
                "color_key": color_key,
            }
        )

    return items


def _draw_items(
    axis: Axes,
    items: list[dict[str, Any]],
    colors: ColorMap,
    linestyle: str,
    prefix: str | None = None,
) -> None:
    for item in items:
        x1, y1, x2, y2 = item["box"]
        color = colors.get(item["label_id"], colors[item["color_key"]])
        rect = plt.Rectangle(
            (float(x1), float(y1)),
            float(x2 - x1),
            float(y2 - y1),
            fill=False,
            edgecolor=color,
            linewidth=2,
            linestyle=linestyle,
        )
        axis.add_patch(rect)

        label = item["label"] if prefix is None else f"{prefix}: {item['label']}"
        axis.text(
            float(x1),
            float(y1) - 5,
            label,
            color="white",
            bbox={"facecolor": color, "alpha": 0.75},
        )


def _build_colors(items: list[ColorKey]) -> dict[ColorKey, Color]:
    if not items:
        return {}

    unique_items = list(dict.fromkeys(items))
    generated_colors = distinctipy.get_colors(len(unique_items), rng=42)
    return dict(zip(unique_items, generated_colors))


def _resolve_colors(
    items: list[dict[str, Any]],
    colors: ColorMap | None,
) -> dict[ColorKey, Color]:
    colors_by_key = _build_colors(items=[item["color_key"] for item in items])
    if colors is not None:
        colors_by_key.update(colors)
    return colors_by_key


def _prediction_title(prediction: DetectionPrediction) -> str:
    image_id = prediction.image_id
    num_predictions = prediction.num_predictions
    if image_id is None:
        return f"Predictions: {num_predictions}"
    return f"Image ID: {image_id}\nPredictions: {num_predictions}"


def _target_prediction_title(target: DetectionTarget, prediction: DetectionPrediction) -> str:
    image_id = target.get("image_id", prediction.image_id)
    num_targets = len(target["boxes"])
    num_predictions = prediction.num_predictions
    if image_id is None:
        return f"Targets: {num_targets}, Predictions: {num_predictions}"
    return f"Image ID: {image_id}\nTargets: {num_targets}, Predictions: {num_predictions}"
