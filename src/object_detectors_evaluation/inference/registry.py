from typing import TypeAlias

from object_detectors_evaluation.inference.engines.base import BaseDetectionInferenceEngine
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

    if model_spec.inference_engine is not None:
        return get_detection_inference_engine_class(name=model_spec.inference_engine)

    msg = (
        f"Model `{model_spec.name}` has no default inference engine. "
        "Pass `engine_name` explicitly or set `inference_engine` in the model spec."
    )
    logger.error(msg)
    raise ValueError(msg)
