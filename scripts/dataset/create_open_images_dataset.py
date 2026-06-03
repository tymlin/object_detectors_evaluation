from object_detectors_evaluation.consts import FIFTYONE_DATASETS_DIRPATH
from object_detectors_evaluation.datasets import OpenImagesDataset
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.utils.types import Split

DATASET_DIRPATH = FIFTYONE_DATASETS_DIRPATH / "open-images-v7"
SPLIT: Split = "test"
CLASSES_OF_INTEREST = ["Person"]
INDEX = 0


logger.info(f"Creating dataset `Open Images` from path: '{DATASET_DIRPATH}'")
dataset = OpenImagesDataset(
    dataset_dirpath=DATASET_DIRPATH,
    split=SPLIT,
    classes_of_interest=CLASSES_OF_INTEREST,
)
logger.info(f"Dataset `Open Images` (split: {SPLIT}) created with {len(dataset)} samples")
logger.info(f"Dataset `Open Images` classes ({len(dataset.get_class_names())}): {dataset.get_class_names()}")

image, target = dataset[INDEX]
logger.info(f"First `Open Images` sample image size: {image.size}")
logger.info(f"First `Open Images` sample objects: {len(target['labels'])}")
