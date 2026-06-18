import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from object_detectors_evaluation.consts import FIFTYONE_DATASETS_DIRPATH, MODELS_DIRPATH, RESULTS_DIRPATH
from object_detectors_evaluation.datasets import (
    BaseDetectionDataset,
    COCODataset,
    DatasetName,
    DatasetSplit,
    DetectionDatasetConfig,
    OpenImagesDataset,
)
from object_detectors_evaluation.inference import DetectionInferenceConfig, resolve_detection_inference_engine_class
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.models.downloaders import download_model
from object_detectors_evaluation.models.registry import get_detection_model_spec
from object_detectors_evaluation.visualization import plot_prediction, plot_target_prediction

DEFAULT_DATASET_NAME: DatasetName = "coco"
DEFAULT_DATASETS_DIRPATH = FIFTYONE_DATASETS_DIRPATH
DEFAULT_SPLIT: DatasetSplit = "validation"
DEFAULT_CLASSES_OF_INTEREST: list[str] | None = None
DEFAULT_INCLUDE_CROWD = True
DEFAULT_DROP_IMAGES_WITH_CROWD = False
DEFAULT_REMOVE_EMPTY_IMAGES = False
DEFAULT_INDEX = 0

DEFAULT_MODEL_NAME = "yolov8n"
DEFAULT_MODELS_DIRPATH = MODELS_DIRPATH
DEFAULT_ENGINE_NAME: str | None = None
DEFAULT_DEVICE = "auto"
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_MAX_DETECTIONS: int | None = None
DEFAULT_DTYPE: str | None = None
DEFAULT_OVERWRITE_MODEL = False
DEFAULT_TOP_K_LOGGED_PREDICTIONS = 10

DEFAULT_PLOT_PREDICTION = True
DEFAULT_PLOT_TARGET_PREDICTION = True
DEFAULT_SHOW_FIGURES = True
DEFAULT_SAVE_FIGURES = True
DEFAULT_FIGURES_DIRPATH = RESULTS_DIRPATH / "inference_figures"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run object detection inference on one dataset sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-name",
        choices=("coco", "open_images"),
        default=DEFAULT_DATASET_NAME,
        help="Dataset implementation to load.",
    )
    parser.add_argument(
        "--datasets-dirpath",
        type=Path,
        default=DEFAULT_DATASETS_DIRPATH,
        help="Root directory containing FiftyOne datasets.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default=DEFAULT_SPLIT,
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES_OF_INTEREST,
        help="Optional classes to retain.",
    )
    parser.add_argument(
        "--include-crowd",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_INCLUDE_CROWD,
        help="Include crowd annotations in dataset targets.",
    )
    parser.add_argument(
        "--drop-images-with-crowd",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DROP_IMAGES_WITH_CROWD,
        help="Drop dataset images containing crowd annotations.",
    )
    parser.add_argument(
        "--remove-empty-images",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REMOVE_EMPTY_IMAGES,
        help="Remove images without retained annotations.",
    )
    parser.add_argument("--index", type=int, default=DEFAULT_INDEX, help="Dataset sample index.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Registered detection model name.")
    parser.add_argument(
        "--models-dirpath",
        type=Path,
        default=DEFAULT_MODELS_DIRPATH,
        help="Root model directory.",
    )
    parser.add_argument("--engine-name", default=DEFAULT_ENGINE_NAME, help="Optional inference engine override.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Inference device.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help="Prediction score threshold.",
    )

    max_detections_group = parser.add_mutually_exclusive_group()
    max_detections_group.add_argument(
        "--max-detections",
        type=int,
        dest="max_detections",
        help="Maximum detections returned for the image.",
    )
    max_detections_group.add_argument(
        "--no-max-detections",
        action="store_const",
        const=None,
        dest="max_detections",
        help="Do not limit the number of returned detections.",
    )
    parser.set_defaults(max_detections=DEFAULT_MAX_DETECTIONS)

    dtype_group = parser.add_mutually_exclusive_group()
    dtype_group.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        dest="dtype",
        help="Inference dtype.",
    )
    dtype_group.add_argument(
        "--no-dtype",
        action="store_const",
        const=None,
        dest="dtype",
        help="Do not request a specific inference dtype.",
    )
    parser.set_defaults(dtype=DEFAULT_DTYPE)

    parser.add_argument(
        "--overwrite-model",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OVERWRITE_MODEL,
        help="Overwrite an existing direct-URL model artifact.",
    )
    parser.add_argument(
        "--top-k-logged-predictions",
        type=int,
        default=DEFAULT_TOP_K_LOGGED_PREDICTIONS,
        help="Maximum number of predictions written to logs.",
    )
    parser.add_argument(
        "--plot-prediction",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_PLOT_PREDICTION,
        help="Create the prediction visualization.",
    )
    parser.add_argument(
        "--plot-target-prediction",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_PLOT_TARGET_PREDICTION,
        help="Create the combined target and prediction visualization.",
    )
    parser.add_argument(
        "--show-figures",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SHOW_FIGURES,
        help="Display generated figures.",
    )
    parser.add_argument(
        "--save-figures",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_FIGURES,
        help="Save generated figures.",
    )
    parser.add_argument(
        "--figures-dirpath",
        type=Path,
        default=DEFAULT_FIGURES_DIRPATH,
        help="Directory where generated figures are saved.",
    )
    return parser.parse_args()


def create_dataset(
    dataset_name: DatasetName,
    datasets_dirpath: Path,
    split: DatasetSplit,
    classes_of_interest: list[str] | None,
    include_crowd: bool,
    drop_images_with_crowd: bool,
    remove_empty_images: bool,
) -> BaseDetectionDataset:
    """Create the configured detection dataset.

    :param dataset_name: Dataset implementation name.
    :param datasets_dirpath: Root directory containing downloaded datasets.
    :param split: Dataset split to load.
    :param classes_of_interest: Optional class names to retain.
    :param include_crowd: Whether crowd annotations are included in targets.
    :param drop_images_with_crowd: Whether images containing crowd annotations are removed.
    :param remove_empty_images: Whether images without retained annotations are removed.
    :return: Detection dataset instance.
    """
    if dataset_name == "coco":
        dataset_dirpath = datasets_dirpath / "coco-2017"
        logger.info(f"Creating dataset `COCO` (split: {split}) from path: '{dataset_dirpath}'")
        config = DetectionDatasetConfig(
            dataset_dirpath=dataset_dirpath,
            split=split,
            classes_of_interest=classes_of_interest,
            include_crowd=include_crowd,
            drop_images_with_crowd=drop_images_with_crowd,
            remove_empty_images=remove_empty_images,
        )
        return COCODataset(
            config=config,
        )

    if dataset_name == "open_images":
        dataset_dirpath = datasets_dirpath / "open-images-v7"
        logger.info(f"Creating dataset `Open Images` (split: {split}) from path: '{dataset_dirpath}'")
        config = DetectionDatasetConfig(
            dataset_dirpath=dataset_dirpath,
            split=split,
            classes_of_interest=classes_of_interest,
            include_crowd=include_crowd,
            drop_images_with_crowd=drop_images_with_crowd,
            remove_empty_images=remove_empty_images,
        )
        return OpenImagesDataset(
            config=config,
        )

    msg = f"Unsupported dataset `{dataset_name}`"
    logger.error(msg)
    raise ValueError(msg)


def main() -> None:
    args = parse_args()
    dataset = create_dataset(
        dataset_name=args.dataset_name,
        datasets_dirpath=args.datasets_dirpath,
        split=args.split,
        classes_of_interest=args.classes,
        include_crowd=args.include_crowd,
        drop_images_with_crowd=args.drop_images_with_crowd,
        remove_empty_images=args.remove_empty_images,
    )
    logger.info(f"Dataset `{args.dataset_name}` created with {len(dataset)} samples")
    logger.info(
        f"Dataset `{args.dataset_name}` classes ({len(dataset.get_class_names())}): {dataset.get_class_names()}"
    )

    image, target = dataset[args.index]
    image_id = target["image_id"]
    logger.info(f"Loaded dataset sample `{args.index}` with image id `{image_id}` and {len(target['labels'])} targets")

    model_spec = get_detection_model_spec(args.model_name)
    logger.info(f"Preparing model `{model_spec.name}` from path: '{args.models_dirpath}'")
    download_kwargs = {"overwrite": args.overwrite_model} if model_spec.source_type == "url" else {}
    model_artifact = download_model(
        spec=model_spec,
        models_dirpath=args.models_dirpath,
        **download_kwargs,
    )
    logger.info(f"Model `{model_artifact.spec.name}` ready at path: '{model_artifact.dirpath}'")

    engine_class = resolve_detection_inference_engine_class(model_spec=model_spec, engine_name=args.engine_name)
    config = DetectionInferenceConfig(
        device=args.device,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        dtype=args.dtype,
    )
    engine = engine_class(model_artifact=model_artifact, config=config)
    image_array = np.asarray(image)
    prediction_batch = engine(images=[image_array], image_ids=[image_id])
    prediction = prediction_batch.predictions[0]

    logger.info(
        f"Model `{model_spec.name}` predicted {prediction.num_predictions} objects "
        f"for image id `{prediction.image_id}`"
    )
    if prediction_batch.latency is not None:
        latency = prediction_batch.latency
        logger.info(
            f"Latency for model `{model_spec.name}`: \n"
            f"\tpreprocess {latency.preprocess_ms:.2f} ms, \n"
            f"\tinference {latency.inference_ms:.2f} ms, \n"
            f"\tpostprocess {latency.postprocess_ms:.2f} ms, \n"
            f"\ttotal {latency.total_ms:.2f} ms"
        )

    for idx in range(min(args.top_k_logged_predictions, prediction.num_predictions)):
        label_name = None if prediction.labels_names is None else prediction.labels_names[idx]
        logger.info(
            f"Prediction {idx}: label `{prediction.labels[idx]}`, name `{label_name}`, "
            f"score `{prediction.scores[idx]:.4f}`, box `{prediction.boxes[idx].tolist()}`"
        )

    if args.plot_prediction:
        fig = plot_prediction(
            image=image_array,
            prediction=prediction,
            score_threshold=args.score_threshold,
        )
        _handle_figure(
            fig=fig,
            filename=f"{args.model_name}_{args.dataset_name}_{image_id}_prediction.png",
            figures_dirpath=args.figures_dirpath,
            save_figures=args.save_figures,
        )

    if args.plot_target_prediction:
        fig = plot_target_prediction(
            image=image_array,
            target=target,
            prediction=prediction,
            score_threshold=args.score_threshold,
        )
        _handle_figure(
            fig=fig,
            filename=f"{args.model_name}_{args.dataset_name}_{image_id}_target_prediction.png",
            figures_dirpath=args.figures_dirpath,
            save_figures=args.save_figures,
        )

    if args.show_figures:
        plt.show(block=True)

    plt.close("all")


def _handle_figure(
    fig: Figure,
    filename: str,
    figures_dirpath: Path,
    save_figures: bool,
) -> None:
    """Save a visualization figure according to script settings.

    :param fig: Matplotlib figure.
    :param filename: Output filename used when figure saving is enabled.
    :param figures_dirpath: Directory where figures are saved.
    :param save_figures: Whether the figure is saved.
    """
    if save_figures:
        figures_dirpath.mkdir(parents=True, exist_ok=True)
        filepath = figures_dirpath / filename
        logger.info(f"Saving inference figure to path: '{filepath}'")
        fig.savefig(filepath, bbox_inches="tight")


if __name__ == "__main__":
    main()
