import argparse
from pathlib import Path

from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.reports import generate_detection_evaluation_report

DEFAULT_OUTPUT_DIRPATH: Path | None = None
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_TOP_CLASSES = 20
DEFAULT_MODEL_NAMES: list[str] | None = None
DEFAULT_EXCLUDE_MODEL_NAMES: list[str] | None = None
DEFAULT_TOP_MODELS: int | None = None
DEFAULT_LATENCY_FIELD = "inference_mean_ms"
DEFAULT_OVERWRITE = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate a static report for a completed detection evaluation run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dirpath",
        type=Path,
        required=True,
        help="Completed detection evaluation run directory.",
    )
    parser.add_argument(
        "--output-dirpath",
        type=Path,
        default=DEFAULT_OUTPUT_DIRPATH,
        help="Report output directory. Defaults to <run>/report.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help="Common score threshold for prediction-derived plots.",
    )
    parser.add_argument(
        "--top-classes",
        type=int,
        default=DEFAULT_TOP_CLASSES,
        help="Number of frequent target classes shown in the per-class heatmap.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODEL_NAMES,
        help="Optional model names to include.",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="+",
        default=DEFAULT_EXCLUDE_MODEL_NAMES,
        help="Optional model names to exclude.",
    )
    parser.add_argument(
        "--top-models",
        type=int,
        default=DEFAULT_TOP_MODELS,
        help="Optional number of highest-mAP models to include.",
    )
    parser.add_argument(
        "--latency-field",
        choices=("inference_mean_ms", "total_mean_ms"),
        default=DEFAULT_LATENCY_FIELD,
        help="Latency field used for the accuracy-versus-latency plot.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OVERWRITE,
        help="Replace an existing report output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(f"Generating detection report for run at path: '{args.run_dirpath}'")
    report_filepath = generate_detection_evaluation_report(
        run_dirpath=args.run_dirpath,
        output_dirpath=args.output_dirpath,
        score_threshold=args.score_threshold,
        top_classes=args.top_classes,
        model_names=args.models,
        exclude_model_names=args.exclude_models,
        top_models=args.top_models,
        latency_field=args.latency_field,
        overwrite=args.overwrite,
    )
    logger.info(f"Detection report saved to path: '{report_filepath}'")


if __name__ == "__main__":
    main()
