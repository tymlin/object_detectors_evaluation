import argparse
from pathlib import Path

from object_detectors_evaluation.consts import MODELS_DIRPATH
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.models import ModelSpec
from object_detectors_evaluation.models.downloaders import download_model
from object_detectors_evaluation.models.registry import DETECTION_MODELS

DEFAULT_MODEL_SPECS: tuple[ModelSpec, ...] = DETECTION_MODELS
DEFAULT_MODELS_DIRPATH = MODELS_DIRPATH
DEFAULT_OVERWRITE = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download all registered detection models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models-dirpath",
        type=Path,
        default=DEFAULT_MODELS_DIRPATH,
        help="Root model directory.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OVERWRITE,
        help="Overwrite existing direct-URL model artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(f"Downloading {len(DEFAULT_MODEL_SPECS)} detection models to path: '{args.models_dirpath}'")

    for idx, model_spec in enumerate(DEFAULT_MODEL_SPECS, start=1):
        logger.info(f"Downloading model {idx}/{len(DEFAULT_MODEL_SPECS)}: `{model_spec.name}`")
        download_kwargs = {"overwrite": args.overwrite} if model_spec.source_type == "url" else {}
        downloaded_model = download_model(
            spec=model_spec,
            models_dirpath=args.models_dirpath,
            **download_kwargs,
        )
        logger.info(
            f"Downloaded model `{downloaded_model.spec.name}` with {len(downloaded_model.filepaths)} files "
            f"to path: '{downloaded_model.dirpath}'"
        )


if __name__ == "__main__":
    main()
