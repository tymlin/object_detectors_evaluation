from object_detectors_evaluation.consts import CONFIGS_DIRPATH
from object_detectors_evaluation.evaluation import DetectionEvaluator
from object_detectors_evaluation.loggers import logger

CONFIG_FILEPATH = CONFIGS_DIRPATH / "detection_evaluator_coco.yaml"


def main() -> None:
    """Run detection evaluator from a YAML config file."""
    logger.info(f"Running detection evaluator from config path: '{CONFIG_FILEPATH}'")
    evaluator = DetectionEvaluator.from_yaml(filepath=CONFIG_FILEPATH)
    result = evaluator.evaluate()
    logger.info(f"Detection evaluator run `{result.run_name}` saved to path: '{result.run_dirpath}'")


if __name__ == "__main__":
    main()
