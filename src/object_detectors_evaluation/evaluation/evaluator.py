import json
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from natsort import natsorted
from PIL import Image

from object_detectors_evaluation.consts import MODELS_DIRPATH, NOW, RUNS_DIRPATH
from object_detectors_evaluation.datasets import BaseDetectionDataset, COCODataset, OpenImagesDataset
from object_detectors_evaluation.datasets.fiftyone import download_coco_dataset, download_open_images_dataset
from object_detectors_evaluation.datasets.types import DetectionTarget
from object_detectors_evaluation.evaluation.configs import DetectionEvaluatorConfig, DetectionEvaluatorModelConfig
from object_detectors_evaluation.evaluation.metrics import DetectionMeanAveragePrecision, to_jsonable
from object_detectors_evaluation.evaluation.results import (
    DetectionEvaluationLatencySummary,
    DetectionEvaluationModelResult,
    DetectionEvaluationRunResult,
)
from object_detectors_evaluation.inference.engines import BaseDetectionInferenceEngine
from object_detectors_evaluation.inference.predictions import DetectionLatency, DetectionPrediction
from object_detectors_evaluation.inference.registry import resolve_detection_inference_engine_class
from object_detectors_evaluation.inference.types import ImageInput
from object_detectors_evaluation.loggers import configure_logger, logger
from object_detectors_evaluation.models import ModelArtifact, ModelSpec
from object_detectors_evaluation.models.downloaders import build_downloaded_model, download_model, get_model_dirpath
from object_detectors_evaluation.models.registry import get_detection_model_spec
from object_detectors_evaluation.visualization import plot_prediction, plot_target_prediction


class DetectionEvaluator:
    """Evaluate one dataset against one or more detection models.

    :param config: Detection evaluator config.
    :param config_filepath: Optional source YAML filepath.
    """

    run_group_name = "detectors_eval"

    def __init__(
        self,
        config: DetectionEvaluatorConfig,
        config_filepath: str | Path | None = None,
    ) -> None:
        self.config = config
        self.config_filepath = Path(config_filepath) if config_filepath is not None else None
        self.run_name = NOW
        self.run_dirpath = RUNS_DIRPATH / self.run_group_name / self.run_name
        self.dataset_config = config.dataset.config
        self.run_dirpath.mkdir(parents=True, exist_ok=True)
        configure_logger(
            run_dirpath=self.run_dirpath / "logs",
            log_filename="evaluation.log",
        )
        logger.info(f"Initializing detection evaluator with config: {config.model_dump_json(indent=4)}")
        logger.info(f"Initialized detection evaluator run `{self.run_name}` at path: '{self.run_dirpath}'")

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> "DetectionEvaluator":
        """Create evaluator from a YAML config file.

        :param filepath: YAML config filepath.
        :return: Detection evaluator.
        """
        config = DetectionEvaluatorConfig.from_yaml(filepath=filepath)
        return cls(config=config, config_filepath=filepath)

    def evaluate(self) -> DetectionEvaluationRunResult:
        """Run evaluation for all configured models.

        :return: Multi-model evaluation result.
        """
        self._save_run_config()
        dataset = self._create_dataset()
        model_results = []

        for model_config in self.config.models:
            model_results.append(
                self._evaluate_model(
                    model_config=model_config,
                    dataset=dataset,
                )
            )

        result = DetectionEvaluationRunResult(
            run_name=self.run_name,
            run_dirpath=str(self.run_dirpath),
            dataset_name=self.config.dataset.name,
            dataset_split=self.dataset_config.split,
            num_samples=len(dataset),
            models=tuple(model_results),
        )
        self._save_json(filepath=self.run_dirpath / "summary.json", data=result.model_dump(mode="json"))
        logger.info(f"Finished detection evaluator run `{self.run_name}` at path: '{self.run_dirpath}'")
        return result

    def _create_dataset(self) -> BaseDetectionDataset:
        if self.config.dataset.auto_download:
            self._download_dataset()

        logger.info(f"Creating evaluation dataset `{self.config.dataset.name}`")
        if self.config.dataset.name == "coco":
            return COCODataset(config=self.dataset_config)

        if self.config.dataset.name == "open_images":
            return OpenImagesDataset(config=self.dataset_config)

        msg = f"Unsupported evaluation dataset `{self.config.dataset.name}`"
        logger.error(msg)
        raise ValueError(msg)

    def _download_dataset(self) -> None:
        dataset_root_dirpath = self.dataset_config.dataset_dirpath.parent
        if self.config.dataset.name == "coco":
            download_coco_dataset(
                dataset_dirpath=dataset_root_dirpath,
                split=self.dataset_config.split,
                classes=self.dataset_config.classes_of_interest,
            )
            return

        if self.config.dataset.name == "open_images":
            download_open_images_dataset(
                dataset_dirpath=dataset_root_dirpath,
                split=self.dataset_config.split,
                classes=self.dataset_config.classes_of_interest,
            )
            return

        msg = f"Unsupported evaluation dataset download for `{self.config.dataset.name}`"
        logger.error(msg)
        raise ValueError(msg)

    def _evaluate_model(
        self,
        model_config: DetectionEvaluatorModelConfig,
        dataset: BaseDetectionDataset,
    ) -> DetectionEvaluationModelResult:
        model_spec = get_detection_model_spec(model_config.name)
        self._validate_class_space(model_class_space=model_spec.class_space)
        model_artifact = self._get_model_artifact(model_spec=model_spec, auto_download=model_config.auto_download)

        engine_class = resolve_detection_inference_engine_class(
            model_spec=model_spec,
            engine_name=model_config.engine_name,
        )
        engine = engine_class(model_artifact=model_artifact, config=model_config.config)
        model_dirpath = self.run_dirpath / "models" / model_spec.name
        model_dirpath.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting warmup for model `{model_spec.name}`")
        self._warmup_model(engine=engine, dataset=dataset)
        logger.info(f"Finished warmup for model `{model_spec.name}`")

        metric = DetectionMeanAveragePrecision(config=self.config.evaluation.metrics)
        latencies = []
        predictions_filepath = model_dirpath / "predictions.jsonl"
        num_plotted_samples = 0

        for batch_index, batch in enumerate(self._iter_batches(dataset=dataset)):
            images, targets = batch
            image_ids = [target["image_id"] for target in targets]
            prepared_images = [self._prepare_image(image=image) for image in images]
            prediction_batch = engine(images=prepared_images, image_ids=image_ids)
            predictions = list(prediction_batch.predictions)
            metric.update(predictions=predictions, targets=targets)
            if prediction_batch.latency is not None:
                latencies.append(prediction_batch.latency)
            if self.config.outputs.save_predictions:
                self._append_predictions(filepath=predictions_filepath, predictions=predictions)
            if self._should_save_plots(num_plotted_samples=num_plotted_samples):
                num_plotted_samples = self._save_plots(
                    model_config=model_config,
                    model_dirpath=model_dirpath,
                    batch_index=batch_index,
                    images=prepared_images,
                    targets=targets,
                    predictions=predictions,
                    num_plotted_samples=num_plotted_samples,
                )

        metrics = metric.compute()
        latency_summary = self._summarize_latencies(latencies=latencies)
        self._save_json(filepath=model_dirpath / "metrics.json", data=metrics)
        self._save_json(filepath=model_dirpath / "latency.json", data=latency_summary.model_dump(mode="json"))

        result = DetectionEvaluationModelResult(
            model_name=model_spec.name,
            engine_name=engine.engine_name,
            metrics=metrics,
            latency=latency_summary,
            num_samples=len(dataset),
            run_dirpath=str(model_dirpath),
        )
        logger.info(f"Finished evaluating model `{model_spec.name}`")
        return result

    def _get_model_artifact(self, model_spec: ModelSpec, auto_download: bool) -> ModelArtifact:
        if auto_download:
            return download_model(
                spec=model_spec,
                models_dirpath=MODELS_DIRPATH,
            )

        model_dirpath = get_model_dirpath(spec=model_spec, models_dirpath=MODELS_DIRPATH)
        if not model_dirpath.is_dir():
            msg = f"Could not find local model directory for `{model_spec.name}` at path: '{model_dirpath}'"
            logger.error(msg)
            raise FileNotFoundError(msg)

        filepaths = tuple(natsorted(model_dirpath.rglob("*")))
        filepaths = tuple(filepath for filepath in filepaths if filepath.is_file())
        return build_downloaded_model(spec=model_spec, dirpath=model_dirpath, filepaths=filepaths)

    def _warmup_model(self, engine: BaseDetectionInferenceEngine, dataset: BaseDetectionDataset) -> None:
        if self.config.evaluation.warmup_iterations == 0 or len(dataset) == 0:
            return

        image, target = dataset[0]
        prepared_image = self._prepare_image(image=image)
        image_id = target["image_id"]
        for _ in range(self.config.evaluation.warmup_iterations):
            engine(images=[prepared_image], image_ids=[image_id])

    def _iter_batches(self, dataset: BaseDetectionDataset) -> Iterator[tuple[list[object], list[DetectionTarget]]]:
        batch_size = self.config.evaluation.batch_size
        for start_index in range(0, len(dataset), batch_size):
            batch_items = [dataset[index] for index in range(start_index, min(start_index + batch_size, len(dataset)))]
            images, targets = zip(*batch_items)
            yield list(images), list(targets)

    def _save_plots(
        self,
        model_config: DetectionEvaluatorModelConfig,
        model_dirpath: Path,
        batch_index: int,
        images: list[ImageInput],
        targets: list[DetectionTarget],
        predictions: list[DetectionPrediction],
        num_plotted_samples: int,
    ) -> int:
        plot_dirpath = model_dirpath / "plots"
        plot_dirpath.mkdir(parents=True, exist_ok=True)
        plot_score_threshold = self.config.outputs.plot_score_threshold
        if plot_score_threshold is None:
            plot_score_threshold = model_config.config.score_threshold

        for item_index, (image, target, prediction) in enumerate(zip(images, targets, predictions)):
            if not self._should_save_plots(num_plotted_samples=num_plotted_samples):
                return num_plotted_samples

            sample_id = f"batch_{batch_index:06d}_sample_{item_index:02d}"
            prediction_fig = plot_prediction(
                image=image,
                prediction=prediction,
                score_threshold=plot_score_threshold,
            )
            prediction_fig.savefig(plot_dirpath / f"{sample_id}_prediction.png", bbox_inches="tight")
            plt.close(prediction_fig)

            target_prediction_fig = plot_target_prediction(
                image=image,
                target=target,
                prediction=prediction,
                score_threshold=plot_score_threshold,
            )
            target_prediction_fig.savefig(
                plot_dirpath / f"{sample_id}_target_prediction.png",
                bbox_inches="tight",
            )
            plt.close(target_prediction_fig)
            num_plotted_samples += 1

        return num_plotted_samples

    def _should_save_plots(self, num_plotted_samples: int) -> bool:
        if not self.config.outputs.save_plots:
            return False
        if self.config.outputs.num_plot_samples is None:
            return True
        return num_plotted_samples < self.config.outputs.num_plot_samples

    def _validate_class_space(self, model_class_space: str | None) -> None:
        if self.config.dataset.name == model_class_space:
            return

        msg = (
            f"Evaluator currently supports matching class spaces only, got dataset `{self.config.dataset.name}` "
            f"and model class space `{model_class_space}`"
        )
        logger.error(msg)
        raise ValueError(msg)

    def _save_run_config(self) -> None:
        with (self.run_dirpath / "config.yaml").open("w") as file:
            yaml.safe_dump(self.config.model_dump(mode="json"), file, sort_keys=False)

        resolved_config = self.config.model_dump(mode="json")
        resolved_config["run_name"] = self.run_name
        resolved_config["run_dirpath"] = str(self.run_dirpath)
        resolved_config["dataset"]["config"]["dataset_dirpath"] = str(self.dataset_config.dataset_dirpath)
        self._save_json(filepath=self.run_dirpath / "resolved_config.json", data=resolved_config)

    def _append_predictions(self, filepath: Path, predictions: list[DetectionPrediction]) -> None:
        with filepath.open("a") as file:
            for prediction in predictions:
                file.write(json.dumps(to_jsonable(value=prediction.model_dump(mode="python"))) + "\n")

    @staticmethod
    def _prepare_image(image: object) -> ImageInput:
        if isinstance(image, torch.Tensor):
            return image
        if isinstance(image, Image.Image):
            return np.asarray(image)
        if isinstance(image, np.ndarray):
            return image

        msg = f"Unsupported evaluator image type `{type(image).__name__}`"
        logger.error(msg)
        raise TypeError(msg)

    @staticmethod
    def _summarize_latencies(latencies: list[DetectionLatency]) -> DetectionEvaluationLatencySummary:
        if not latencies:
            return DetectionEvaluationLatencySummary(
                preprocess_mean_ms=0.0,
                inference_mean_ms=0.0,
                postprocess_mean_ms=0.0,
                total_mean_ms=0.0,
                num_batches=0,
            )

        preprocess_mean_ms = float(np.mean([latency.preprocess_ms for latency in latencies]))
        inference_mean_ms = float(np.mean([latency.inference_ms for latency in latencies]))
        postprocess_mean_ms = float(np.mean([latency.postprocess_ms for latency in latencies]))
        total_mean_ms = float(np.mean([latency.total_ms for latency in latencies]))
        return DetectionEvaluationLatencySummary(
            preprocess_mean_ms=preprocess_mean_ms,
            inference_mean_ms=inference_mean_ms,
            postprocess_mean_ms=postprocess_mean_ms,
            total_mean_ms=total_mean_ms,
            num_batches=len(latencies),
        )

    @staticmethod
    def _save_json(filepath: Path, data: object) -> None:
        logger.info(f"Saving JSON file to path: '{filepath}'")
        with filepath.open("w") as file:
            json.dump(data, file, indent=4)
