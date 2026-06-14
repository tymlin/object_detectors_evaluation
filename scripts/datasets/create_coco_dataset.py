from object_detectors_evaluation.consts import FIFTYONE_DATASETS_DIRPATH
from object_detectors_evaluation.datasets import COCODataset, DatasetSplit, DetectionDatasetConfig
from object_detectors_evaluation.loggers import logger

DATASET_DIRPATH = FIFTYONE_DATASETS_DIRPATH / "coco-2017"
SPLIT: DatasetSplit = "validation"
CLASSES_OF_INTEREST = ["person"]
INCLUDE_CROWD = True
DROP_IMAGES_WITH_CROWD = False
REMOVE_EMPTY_IMAGES = False
INDEX = 0


logger.info(f"Creating dataset `COCO` (split: {SPLIT}) from path: '{DATASET_DIRPATH}'")
config = DetectionDatasetConfig(
    dataset_dirpath=DATASET_DIRPATH,
    split=SPLIT,
    classes_of_interest=CLASSES_OF_INTEREST,
    include_crowd=INCLUDE_CROWD,
    drop_images_with_crowd=DROP_IMAGES_WITH_CROWD,
    remove_empty_images=REMOVE_EMPTY_IMAGES,
)
dataset = COCODataset(
    config=config,
)
logger.info(f"Dataset `COCO` created with {len(dataset)} samples")
logger.info(f"Dataset `COCO` classes ({len(dataset.get_class_names())}): {dataset.get_class_names()}")

image, target = dataset[INDEX]
logger.info(f"First `COCO` sample image size: {image.size}")
logger.info(f"First `COCO` sample objects: {len(target['labels'])}")
