import torch

from object_detectors_evaluation.loggers import logger


def resolve_torch_device(device: str) -> torch.device:
    """Resolve a device string to a torch device.

    :param device: Device string or ``auto``.
    :return: Resolved torch device.
    """
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def resolve_torch_dtype(dtype: str | None) -> torch.dtype | None:
    """Resolve a dtype string to a torch dtype.

    :param dtype: Optional dtype string.
    :return: Resolved torch dtype or ``None``.
    """
    if dtype in (None, "auto"):
        return None

    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtype_by_name:
        msg = f"Torch dtype `{dtype}` is not supported"
        logger.error(msg)
        raise ValueError(msg)

    return dtype_by_name[dtype]
