from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.models import ModelSpec
from object_detectors_evaluation.models.registry import DETECTION_MODELS

MODEL_SPECS: tuple[ModelSpec, ...] = DETECTION_MODELS


def format_model_spec(model_spec: ModelSpec) -> str:
    """Format a model spec for logging.

    :param model_spec: Model spec to format.
    :return: One-line model summary.
    """
    return (
        f"`{model_spec.name}` | family: `{model_spec.family}` | source: `{model_spec.source_type}` | "
        f"training dataset: `{model_spec.training_dataset}` | class space: `{model_spec.class_space}` | "
        f"checkpoint: `{model_spec.checkpoint_format}` | engine: `{model_spec.inference_engine}`"
    )


def main() -> None:
    """List all registered detection models."""
    logger.info(f"Available detection models: {len(MODEL_SPECS)}")

    for idx, model_spec in enumerate(MODEL_SPECS, start=1):
        logger.info(f"{idx}. {format_model_spec(model_spec=model_spec)}")


if __name__ == "__main__":
    main()
