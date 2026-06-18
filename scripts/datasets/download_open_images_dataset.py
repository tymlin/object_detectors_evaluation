import argparse
from pathlib import Path
from typing import Sequence

from object_detectors_evaluation.consts import FIFTYONE_DATASETS_DIRPATH
from object_detectors_evaluation.datasets import DatasetSplit
from object_detectors_evaluation.datasets.fiftyone import download_open_images_dataset
from object_detectors_evaluation.loggers import logger

DEFAULT_DATASET_DIRPATH = FIFTYONE_DATASETS_DIRPATH
DEFAULT_SPLIT: DatasetSplit = "test"
DEFAULT_CLASSES_OF_INTEREST: Sequence[str] | None = None  # ["Person"], ["Person", "Human body", "Man", "Woman"]
DEFAULT_MAX_SAMPLES: int | None = 5


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download the Open Images dataset through FiftyOne.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dirpath",
        type=Path,
        default=DEFAULT_DATASET_DIRPATH,
        help="Directory where FiftyOne stores downloaded datasets.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default=DEFAULT_SPLIT,
        help="Dataset split to download.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES_OF_INTEREST,
        help="Optional class names to download.",
    )
    samples_group = parser.add_mutually_exclusive_group()
    samples_group.add_argument(
        "--max-samples",
        type=int,
        dest="max_samples",
        help="Maximum number of samples to download.",
    )
    samples_group.add_argument(
        "--all-samples",
        action="store_const",
        const=None,
        dest="max_samples",
        help="Download all available samples.",
    )
    parser.set_defaults(max_samples=DEFAULT_MAX_SAMPLES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(
        f"Downloading dataset `Open Images` (split: {args.split}, #samples: {args.max_samples}) "
        f"to path: '{args.dataset_dirpath}'"
    )
    open_images_dataset, open_images_path = download_open_images_dataset(
        dataset_dirpath=args.dataset_dirpath,
        split=args.split,
        classes=args.classes,
        max_samples=args.max_samples,
    )
    logger.info(
        f"Dataset `Open Images` downloaded: {open_images_dataset.num_samples} samples at path: '{open_images_path}'"
    )


if __name__ == "__main__":
    main()
