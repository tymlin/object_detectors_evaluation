from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.models import ModelSpec
from object_detectors_evaluation.models.collections import (
    DFINE_MODELS,
    RFDETR_MODELS,
    RTDETR_MODELS,
    ULTRALYTICS_YOLO_DETECTION_MODELS,
)

DETECTION_MODELS: tuple[ModelSpec, ...] = (
    *DFINE_MODELS,
    *RFDETR_MODELS,
    *RTDETR_MODELS,
    *ULTRALYTICS_YOLO_DETECTION_MODELS,
)

DETECTION_MODEL_NAMES: tuple[str, ...] = tuple(model.name for model in DETECTION_MODELS)
DETECTION_MODEL_BY_NAME: dict[str, ModelSpec] = {model.name: model for model in DETECTION_MODELS}


def get_detection_model_spec(name: str) -> ModelSpec:
    """Get a detection model spec by name.

    :param name: Model name.
    :return: Matching model spec.
    """
    try:
        return DETECTION_MODEL_BY_NAME[name]
    except KeyError as error:
        msg = f"Unknown detection model `{name}`"
        logger.error(msg)
        raise KeyError(msg) from error
