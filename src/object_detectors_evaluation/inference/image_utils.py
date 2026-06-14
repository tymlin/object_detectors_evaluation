from typing import Any, TypeAlias

import numpy as np
import torch

from object_detectors_evaluation.loggers import logger

ImageInput: TypeAlias = np.ndarray | torch.Tensor
ProcessedInputs: TypeAlias = tuple[ImageInput, ...] | dict[str, torch.Tensor]


def validate_image_input(image: Any) -> ImageInput:
    """Validate an image input supported by inference engines.

    :param image: Candidate image input.
    :return: Validated image input.
    """
    if isinstance(image, (np.ndarray, torch.Tensor)):
        return image

    msg = f"Image input must be a NumPy array or torch tensor, got `{type(image).__name__}`"
    logger.error(msg)
    raise TypeError(msg)


def get_image_size(image: ImageInput) -> tuple[int, int]:
    """Return image size as ``(height, width)``.

    :param image: NumPy array or torch tensor image.
    :return: Image size as ``(height, width)``.
    """
    shape = tuple(image.shape)
    if len(shape) < 2:
        msg = f"Image input must have at least 2 dimensions, got shape `{shape}`"
        logger.error(msg)
        raise ValueError(msg)

    if len(shape) == 2:
        height, width = shape
        return int(height), int(width)

    if len(shape) == 3 and shape[0] in (1, 3, 4):
        height, width = shape[1], shape[2]
        return int(height), int(width)

    height, width = shape[0], shape[1]
    return int(height), int(width)
