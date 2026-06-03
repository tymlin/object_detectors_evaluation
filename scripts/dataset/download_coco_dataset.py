from object_detectors_evaluation.consts import FIFTYONE_DATASETS_DIRPATH
from object_detectors_evaluation.datasets.fiftyone import download_coco_dataset
from object_detectors_evaluation.loggers import logger

MAX_SAMPLES: int | None = 5


def main() -> None:
    logger.info(f"Downloading tiny dataset `COCO` to path: '{FIFTYONE_DATASETS_DIRPATH}'")
    coco_dataset, coco_path = download_coco_dataset(
        split="validation",
        classes=["person"],
        max_samples=MAX_SAMPLES,
    )
    logger.info(f"Dataset `COCO` downloaded: {coco_dataset.num_samples} samples at path: '{coco_path}'")


if __name__ == "__main__":
    main()
