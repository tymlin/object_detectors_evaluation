from object_detectors_evaluation.consts import DATASETS_DIRPATH
from object_detectors_evaluation.datasets.fiftyone import (
    download_open_images_dataset,
)
from object_detectors_evaluation.loggers import logger

FIFTYONE_DATASETS_DIRPATH = DATASETS_DIRPATH / "fiftyone"
MAX_SAMPLES: int | None = 5


def main() -> None:
    logger.info("Downloading tiny Open Images sample to {}", FIFTYONE_DATASETS_DIRPATH)
    open_images_dataset, open_images_path = download_open_images_dataset(
        split="test",
        classes=["Person"],
        max_samples=MAX_SAMPLES,
    )
    logger.info("Open Images downloaded: {} samples at {}", open_images_dataset.num_samples, open_images_path)


if __name__ == "__main__":
    main()
