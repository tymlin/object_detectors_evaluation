import argparse
from pathlib import Path

from object_detectors_evaluation.consts import CONFIGS_DIRPATH
from object_detectors_evaluation.evaluation import DetectionEvaluator
from object_detectors_evaluation.loggers import logger

DEFAULT_CONFIG_FILEPATH = CONFIGS_DIRPATH / "detection_evaluator_coco_all_models.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run detection evaluation from a YAML config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-filepath",
        type=Path,
        default=DEFAULT_CONFIG_FILEPATH,
        help="Detection evaluator YAML config filepath.",
    )
    return parser.parse_args()


def main() -> None:
    """Run detection evaluator from a YAML config file."""
    args = parse_args()
    logger.info(f"Running detection evaluator from config path: '{args.config_filepath}'")
    evaluator = DetectionEvaluator.from_yaml(filepath=args.config_filepath)
    result = evaluator.evaluate()
    logger.info(f"Detection evaluator run `{result.run_name}` saved to path: '{result.run_dirpath}'")


if __name__ == "__main__":
    main()
