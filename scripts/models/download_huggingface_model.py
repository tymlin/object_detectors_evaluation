import argparse
from pathlib import Path

from object_detectors_evaluation.consts import MODELS_DIRPATH
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.models.downloaders import download_huggingface_model
from object_detectors_evaluation.models.registry import get_detection_model_spec

DEFAULT_MODEL_NAME = "rtdetr_v2_r101vd"
DEFAULT_MODELS_DIRPATH = MODELS_DIRPATH


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download a registered Hugging Face detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Registered Hugging Face model name.")
    parser.add_argument(
        "--models-dirpath",
        type=Path,
        default=DEFAULT_MODELS_DIRPATH,
        help="Root model directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_spec = get_detection_model_spec(args.model_name)
    logger.info(f"Downloading Hugging Face model `{model_spec.name}` to path: '{args.models_dirpath}'")
    downloaded_model = download_huggingface_model(
        spec=model_spec,
        models_dirpath=args.models_dirpath,
    )
    logger.info(
        f"Downloaded model `{downloaded_model.spec.name}` with {len(downloaded_model.filepaths)} files "
        f"to path: '{downloaded_model.dirpath}'"
    )


if __name__ == "__main__":
    main()
