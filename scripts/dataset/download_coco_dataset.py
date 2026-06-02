from object_detectors_evaluation.consts import DATASETS_DIRPATH
from object_detectors_evaluation.datasets.fiftyone import (
    download_coco_dataset,
)
from object_detectors_evaluation.loggers import logger

FIFTYONE_DATASETS_DIRPATH = DATASETS_DIRPATH / "fiftyone"
MAX_SAMPLES: int | None = 5


def main() -> None:
    logger.info("Downloading tiny COCO sample to {}", FIFTYONE_DATASETS_DIRPATH)
    coco_dataset, coco_path = download_coco_dataset(
        split="validation",
        classes=["person"],
        max_samples=MAX_SAMPLES,
    )
    logger.info("COCO downloaded: {} samples at {}", coco_dataset.num_samples, coco_path)


if __name__ == "__main__":
    main()
