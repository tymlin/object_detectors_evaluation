from object_detectors_evaluation.consts import FIFTYONE_DATASETS_DIRPATH
from object_detectors_evaluation.datasets.fiftyone import download_open_images_dataset
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.utils.types import Split

SPLIT: Split = "test"
CLASSES_OF_INTEREST = ["Person"]
# CLASSES_OF_INTEREST = ["Person", "Human body", "Man", "Woman"]
MAX_SAMPLES: int | None = 5


def main() -> None:
    logger.info(
        f"Downloading dataset `Open Images` (split: {SPLIT}, #samples: {MAX_SAMPLES}) "
        f"to path: '{FIFTYONE_DATASETS_DIRPATH}'"
    )
    open_images_dataset, open_images_path = download_open_images_dataset(
        split=SPLIT,
        classes=CLASSES_OF_INTEREST,
        max_samples=MAX_SAMPLES,
    )
    logger.info(
        f"Dataset `Open Images` downloaded: {open_images_dataset.num_samples} samples at path: '{open_images_path}'"
    )


if __name__ == "__main__":
    main()
