from object_detectors_evaluation.consts import FIFTYONE_DATASETS_DIRPATH
from object_detectors_evaluation.datasets import DatasetSplit
from object_detectors_evaluation.datasets.fiftyone import download_coco_dataset
from object_detectors_evaluation.loggers import logger

SPLIT: DatasetSplit = "validation"
CLASSES_OF_INTEREST = ["person"]
MAX_SAMPLES: int | None = 5


def main() -> None:
    logger.info(
        f"Downloading dataset `COCO` (split: {SPLIT}, #samples: {MAX_SAMPLES}) "
        f"to path: '{FIFTYONE_DATASETS_DIRPATH}'"
    )
    coco_dataset, coco_path = download_coco_dataset(
        split=SPLIT,
        classes=CLASSES_OF_INTEREST,
        max_samples=MAX_SAMPLES,
    )
    logger.info(f"Dataset `COCO` downloaded: {coco_dataset.num_samples} samples at path: '{coco_path}'")


if __name__ == "__main__":
    main()
