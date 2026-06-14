from typing import Any

import numpy as np
import torch

from object_detectors_evaluation.inference.types import ImageInput
from object_detectors_evaluation.loggers import logger


def validate_image_input(image: Any) -> ImageInput:
    """Validate an image input supported by inference engines.

    :param image: Candidate image input.
    :return: Validated image input.
    """
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[-1] != 3:
            msg = f"NumPy image input must have shape `(H, W, 3)`, got `{tuple(image.shape)}`"
            logger.error(msg)
            raise ValueError(msg)
        return image

    if isinstance(image, torch.Tensor):
        if image.ndim != 3 or image.shape[0] != 3:
            msg = f"Torch image input must have shape `(3, H, W)`, got `{tuple(image.shape)}`"
            logger.error(msg)
            raise ValueError(msg)
        return image

    msg = f"Image input must be a NumPy array or torch tensor, got `{type(image).__name__}`"
    logger.error(msg)
    raise TypeError(msg)


def get_image_size(image: ImageInput) -> tuple[int, int]:
    """Return image size as ``(height, width)``.

    :param image: NumPy array or torch tensor image.
    :return: Image size as ``(height, width)``.
    """
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
        return int(height), int(width)

    height, width = image.shape[1:]
    return int(height), int(width)
