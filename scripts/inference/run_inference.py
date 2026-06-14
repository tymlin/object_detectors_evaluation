from typing import Literal

import numpy as np

from object_detectors_evaluation.consts import FIFTYONE_DATASETS_DIRPATH, MODELS_DIRPATH
from object_detectors_evaluation.datasets import BaseDetectionDataset, COCODataset, OpenImagesDataset
from object_detectors_evaluation.inference import DetectionInferenceConfig, resolve_detection_inference_engine_class
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.models.downloaders import download_model
from object_detectors_evaluation.models.registry import get_detection_model_spec
from object_detectors_evaluation.utils.types import Split

DatasetName = Literal["coco", "open_images"]

DATASET_NAME: DatasetName = "coco"
SPLIT: Split = "validation"
CLASSES_OF_INTEREST: list[str] | None = None
INCLUDE_CROWD = True
DROP_IMAGES_WITH_CROWD = False
REMOVE_EMPTY_IMAGES = False
INDEX = 0

MODEL_NAME = "yolov8n"
# MODEL_NAME = "dfine-small-obj365"
ENGINE_NAME: str | None = None
DEVICE = "auto"
BATCH_SIZE = 1
SCORE_THRESHOLD = 0.25
MAX_DETECTIONS: int | None = None
DTYPE: str | None = None
# DTYPE: str | None = "float32"
OVERWRITE_MODEL = False
TOP_K_LOGGED_PREDICTIONS = 10


def create_dataset() -> BaseDetectionDataset:
    """Create the configured detection dataset.

    :return: Detection dataset instance.
    """
    if DATASET_NAME == "coco":
        dataset_dirpath = FIFTYONE_DATASETS_DIRPATH / "coco-2017"
        logger.info(f"Creating dataset `COCO` (split: {SPLIT}) from path: '{dataset_dirpath}'")
        return COCODataset(
            dataset_dirpath=dataset_dirpath,
            split=SPLIT,
            classes_of_interest=CLASSES_OF_INTEREST,
            include_crowd=INCLUDE_CROWD,
            drop_images_with_crowd=DROP_IMAGES_WITH_CROWD,
            remove_empty_images=REMOVE_EMPTY_IMAGES,
        )

    if DATASET_NAME == "open_images":
        dataset_dirpath = FIFTYONE_DATASETS_DIRPATH / "open-images-v7"
        logger.info(f"Creating dataset `Open Images` (split: {SPLIT}) from path: '{dataset_dirpath}'")
        return OpenImagesDataset(
            dataset_dirpath=dataset_dirpath,
            split=SPLIT,
            classes_of_interest=CLASSES_OF_INTEREST,
            include_crowd=INCLUDE_CROWD,
            drop_images_with_crowd=DROP_IMAGES_WITH_CROWD,
            remove_empty_images=REMOVE_EMPTY_IMAGES,
        )

    msg = f"Unsupported dataset `{DATASET_NAME}`"
    logger.error(msg)
    raise ValueError(msg)


def main() -> None:
    dataset = create_dataset()
    logger.info(f"Dataset `{DATASET_NAME}` created with {len(dataset)} samples")
    logger.info(f"Dataset `{DATASET_NAME}` classes ({len(dataset.get_class_names())}): {dataset.get_class_names()}")

    image, target = dataset[INDEX]
    image_id = target["image_id"]
    logger.info(f"Loaded dataset sample `{INDEX}` with image id `{image_id}` and {len(target['labels'])} targets")

    model_spec = get_detection_model_spec(MODEL_NAME)
    logger.info(f"Preparing model `{model_spec.name}` from path: '{MODELS_DIRPATH}'")
    download_kwargs = {"overwrite": OVERWRITE_MODEL} if model_spec.source_type == "url" else {}
    downloaded_model = download_model(
        spec=model_spec,
        models_dirpath=MODELS_DIRPATH,
        **download_kwargs,
    )
    logger.info(f"Model `{downloaded_model.spec.name}` ready at path: '{downloaded_model.dirpath}'")

    engine_class = resolve_detection_inference_engine_class(model_spec=model_spec, engine_name=ENGINE_NAME)
    config = DetectionInferenceConfig(
        device=DEVICE,
        batch_size=BATCH_SIZE,
        score_threshold=SCORE_THRESHOLD,
        max_detections=MAX_DETECTIONS,
        dtype=DTYPE,
    )
    engine = engine_class(downloaded_model=downloaded_model, config=config)
    image_array = np.asarray(image)
    prediction_batch = engine(images=[image_array], image_ids=[image_id])
    prediction = prediction_batch.predictions[0]

    logger.info(
        f"Model `{model_spec.name}` predicted {prediction.num_predictions} objects "
        f"for image id `{prediction.image_id}`"
    )
    for idx in range(min(TOP_K_LOGGED_PREDICTIONS, prediction.num_predictions)):
        label_name = None if prediction.labels_names is None else prediction.labels_names[idx]
        logger.info(
            f"Prediction {idx}: label `{prediction.labels[idx]}`, name `{label_name}`, "
            f"score `{prediction.scores[idx]:.4f}`, box `{prediction.boxes[idx].tolist()}`"
        )


if __name__ == "__main__":
    main()
