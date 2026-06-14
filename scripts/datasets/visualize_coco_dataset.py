import random

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2

from object_detectors_evaluation.consts import FIFTYONE_DATASETS_DIRPATH
from object_detectors_evaluation.datasets import COCODataset, DatasetSplit, DetectionDatasetConfig
from object_detectors_evaluation.loggers import logger

DATASET_DIRPATH = FIFTYONE_DATASETS_DIRPATH / "coco-2017"
SPLIT: DatasetSplit = "validation"
CLASSES_OF_INTEREST = ["person"]

NUM_SAMPLES = 5
RANDOM_SAMPLES = False
APPLY_TRANSFORMS = False
RANDOM_SEED = 42
INCLUDE_CROWD = True
DROP_IMAGES_WITH_CROWD = False
REMOVE_EMPTY_IMAGES = False

transforms = (
    v2.Compose(
        [
            v2.RandomResizedCrop(size=(800, 800), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.SanitizeBoundingBoxes(),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if APPLY_TRANSFORMS
    else None
)

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
    transforms=transforms,
)
logger.info(f"Dataset `COCO` created with {len(dataset)} samples")

num_samples = min(NUM_SAMPLES, len(dataset))
if RANDOM_SAMPLES:
    random.seed(RANDOM_SEED)
    idxs = random.sample(range(len(dataset)), num_samples)
else:
    idxs = list(range(num_samples))

logger.info(f"Plotting dataset `COCO` indexes: {idxs}")
fig = dataset.plot_images_bbox(idxs, after_transforms=APPLY_TRANSFORMS)
fig.show()
plt.close("all")
