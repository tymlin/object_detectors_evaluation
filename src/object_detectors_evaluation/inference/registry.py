from typing import TypeAlias

from object_detectors_evaluation.inference.base import BaseDetectionInferenceEngine
from object_detectors_evaluation.inference.engines.transformers import TransformersDetectionInferenceEngine
from object_detectors_evaluation.inference.engines.ultralytics import UltralyticsDetectionInferenceEngine
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.models import ModelSpec

DetectionInferenceEngineClass: TypeAlias = type[BaseDetectionInferenceEngine]

DETECTION_INFERENCE_ENGINE_CLASSES: tuple[DetectionInferenceEngineClass, ...] = (
    UltralyticsDetectionInferenceEngine,
    TransformersDetectionInferenceEngine,
)
DETECTION_INFERENCE_ENGINE_BY_NAME: dict[str, DetectionInferenceEngineClass] = {
    engine_class.engine_name: engine_class for engine_class in DETECTION_INFERENCE_ENGINE_CLASSES
}


def get_detection_inference_engine_class(name: str) -> DetectionInferenceEngineClass:
    """Get a detection inference engine class by name.

    :param name: Inference engine name.
    :return: Matching inference engine class.
    """
    try:
        return DETECTION_INFERENCE_ENGINE_BY_NAME[name]
    except KeyError as error:
        msg = f"Unknown detection inference engine `{name}`"
        logger.error(msg)
        raise KeyError(msg) from error


def resolve_detection_inference_engine_class(
    model_spec: ModelSpec,
    engine_name: str | None = None,
) -> DetectionInferenceEngineClass:
    """Resolve the default detection inference engine class for a model spec.

    :param model_spec: Model metadata.
    :param engine_name: Optional explicit inference engine name.
    :return: Resolved inference engine class.
    """
    if engine_name is not None:
        return get_detection_inference_engine_class(name=engine_name)

    for engine_class in DETECTION_INFERENCE_ENGINE_CLASSES:
        if model_spec.checkpoint_format in engine_class.supported_checkpoint_formats:
            return engine_class

    msg = (
        f"Could not resolve detection inference engine for model `{model_spec.name}` "
        f"with checkpoint format `{model_spec.checkpoint_format}`"
    )
    logger.error(msg)
    raise ValueError(msg)
