import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2

from object_detectors_evaluation.consts import FIFTYONE_DATASETS_DIRPATH
from object_detectors_evaluation.datasets import DatasetSplit, DetectionDatasetConfig, OpenImagesDataset
from object_detectors_evaluation.loggers import logger

DEFAULT_DATASET_DIRPATH = FIFTYONE_DATASETS_DIRPATH / "open-images-v7"
DEFAULT_SPLIT: DatasetSplit = "test"
DEFAULT_CLASSES_OF_INTEREST: list[str] | None = None  # ["Person"]
DEFAULT_NUM_SAMPLES = 5
DEFAULT_RANDOM_SAMPLES = False
DEFAULT_APPLY_TRANSFORMS = False
DEFAULT_RANDOM_SEED = 42
DEFAULT_INCLUDE_CROWD = True
DEFAULT_DROP_IMAGES_WITH_CROWD = False
DEFAULT_REMOVE_EMPTY_IMAGES = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Visualize Open Images dataset annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dirpath",
        type=Path,
        default=DEFAULT_DATASET_DIRPATH,
        help="Open Images dataset directory.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default=DEFAULT_SPLIT,
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES_OF_INTEREST,
        help="Optional classes to retain.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of samples to visualize.",
    )
    parser.add_argument(
        "--random-samples",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RANDOM_SAMPLES,
        help="Select random samples instead of the first samples.",
    )
    parser.add_argument(
        "--apply-transforms",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_APPLY_TRANSFORMS,
        help="Apply the example transform pipeline before plotting.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random sampling seed.",
    )
    parser.add_argument(
        "--include-crowd",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_INCLUDE_CROWD,
        help="Include crowd annotations in dataset targets.",
    )
    parser.add_argument(
        "--drop-images-with-crowd",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DROP_IMAGES_WITH_CROWD,
        help="Drop images containing crowd annotations.",
    )
    parser.add_argument(
        "--remove-empty-images",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REMOVE_EMPTY_IMAGES,
        help="Remove images without retained annotations.",
    )
    return parser.parse_args()


def create_transforms() -> v2.Compose:
    """Create the example detection transform pipeline.

    :return: Detection transforms.
    """
    return v2.Compose(
        [
            v2.RandomResizedCrop(size=(800, 800), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.SanitizeBoundingBoxes(),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def main() -> None:
    args = parse_args()
    transforms = create_transforms() if args.apply_transforms else None

    logger.info(f"Creating dataset `Open Images` (split: {args.split}) from path: '{args.dataset_dirpath}'")
    config = DetectionDatasetConfig(
        dataset_dirpath=args.dataset_dirpath,
        split=args.split,
        classes_of_interest=args.classes,
        include_crowd=args.include_crowd,
        drop_images_with_crowd=args.drop_images_with_crowd,
        remove_empty_images=args.remove_empty_images,
    )
    dataset = OpenImagesDataset(
        config=config,
        transforms=transforms,
    )
    logger.info(f"Dataset `Open Images` created with {len(dataset)} samples")

    num_samples = min(args.num_samples, len(dataset))
    logger.info(f"Number of samples to plot: {num_samples}")

    if args.random_samples:
        random_generator = random.Random(args.random_seed)
        idxs = random_generator.sample(range(len(dataset)), num_samples)
    else:
        idxs = list(range(num_samples))

    logger.info(f"Plotting dataset `Open Images` indexes: {idxs}")
    dataset.plot_images_bbox(idxs, after_transforms=args.apply_transforms)
    plt.show(block=True)
    plt.close("all")


if __name__ == "__main__":
    main()
