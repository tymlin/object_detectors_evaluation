from object_detectors_evaluation.consts import FIFTYONE_DATASETS_DIRPATH
from object_detectors_evaluation.datasets import COCODataset
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.utils.types import Split

DATASET_DIRPATH = FIFTYONE_DATASETS_DIRPATH / "coco-2017"
SPLIT: Split = "validation"
CLASSES_OF_INTEREST = ["person"]
INDEX = 0


logger.info(f"Creating dataset `COCO` (split: {SPLIT}) from path: '{DATASET_DIRPATH}'")
dataset = COCODataset(
    dataset_dirpath=DATASET_DIRPATH,
    split=SPLIT,
    classes_of_interest=CLASSES_OF_INTEREST,
)
logger.info(f"Dataset `COCO` created with {len(dataset)} samples")
logger.info(f"Dataset `COCO` classes ({len(dataset.get_class_names())}): {dataset.get_class_names()}")

image, target = dataset[INDEX]
logger.info(f"First `COCO` sample image size: {image.size}")
logger.info(f"First `COCO` sample objects: {len(target['labels'])}")
