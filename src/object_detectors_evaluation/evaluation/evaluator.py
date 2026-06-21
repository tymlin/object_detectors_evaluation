from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from natsort import natsorted
from PIL import Image

from object_detectors_evaluation.consts import MODELS_DIRPATH, NOW, RUNS_DIRPATH
from object_detectors_evaluation.datasets import BaseDetectionDataset, COCODataset, OpenImagesDataset
from object_detectors_evaluation.datasets.fiftyone import download_dataset
from object_detectors_evaluation.datasets.types import DetectionTarget
from object_detectors_evaluation.evaluation.artifacts import (
    DetectionClassArtifact,
    DetectionClassMapArtifact,
    DetectionLatencyArtifact,
    DetectionPredictionArtifact,
    DetectionTargetArtifact,
)
from object_detectors_evaluation.evaluation.configs import (
    DetectionEvaluatorConfig,
    DetectionEvaluatorModelConfig,
    DetectionEvaluatorResolvedConfig,
)
from object_detectors_evaluation.evaluation.metrics import DetectionMeanAveragePrecision, to_jsonable
from object_detectors_evaluation.evaluation.results import (
    DetectionEvaluationLatencySummary,
    DetectionEvaluationModelResult,
    DetectionEvaluationRunResult,
)
from object_detectors_evaluation.inference.engines import BaseDetectionInferenceEngine
from object_detectors_evaluation.inference.predictions import DetectionLatency, DetectionPrediction
from object_detectors_evaluation.inference.registry import resolve_detection_inference_engine_class
from object_detectors_evaluation.inference.types import ImageId, ImageInput
from object_detectors_evaluation.loggers import configure_logger, log_breaking_point, logger
from object_detectors_evaluation.models import ModelArtifact, ModelSpec
from object_detectors_evaluation.models.downloaders import build_downloaded_model, download_model, get_model_dirpath
from object_detectors_evaluation.models.registry import get_detection_model_spec
from object_detectors_evaluation.utils import create_progress
from object_detectors_evaluation.utils.files import append_jsonl, require_dirpath, save_json, save_yaml
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
        self.run_name = NOW if config.run_name_postfix is None else f"{NOW}_{config.run_name_postfix}"
        self.run_dirpath = RUNS_DIRPATH / self.run_group_name / self.run_name
        self.dataset_config = config.dataset.config
        self.run_dirpath.mkdir(parents=True, exist_ok=True)
        configure_logger(
            run_dirpath=self.run_dirpath / "logs",
            log_filename="evaluation.log",
        )
        log_breaking_point(
            logger,
            msg="Detection evaluation run",
            n_top=1,
            n_bottom=1,
            top_char="=",
            bottom_char="=",
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
        num_samples = self._resolve_num_samples(dataset=dataset)
        self._save_class_map(dataset=dataset)
        model_results = []
        num_models_to_evaluate = len(self.config.models)
        for model_ix, model_config in enumerate(self.config.models):
            log_breaking_point(
                logger,
                msg=f"Model: `{model_config.name}` ({model_ix+1}/{num_models_to_evaluate})",
                n_top=1,
                n_bottom=1,
                top_char="-",
                bottom_char="-",
            )
            model_result = self._evaluate_model(
                model_config=model_config,
                dataset=dataset,
                num_samples=num_samples,
                save_targets=model_ix == 0,
            )
            model_results.append(model_result)

        result = DetectionEvaluationRunResult(
            run_name=self.run_name,
            run_dirpath=self.run_dirpath,
            dataset_name=self.config.dataset.name,
            dataset_split=self.dataset_config.split,
            num_samples=num_samples,
            models=tuple(model_results),
        )
        save_json(filepath=self.run_dirpath / "summary.json", data=result.model_dump(mode="json"))
        log_breaking_point(
            logger,
            msg="Finished detection evaluation run",
            n_top=1,
            n_bottom=1,
            top_char="=",
            bottom_char="=",
        )
        logger.info(f"Finished detection evaluator run `{self.run_name}` at path: '{self.run_dirpath}'")
        return result

    def _create_dataset(self) -> BaseDetectionDataset:
        log_breaking_point(
            logger,
            msg=f"Dataset: `{self.config.dataset.name}`",
            n_top=1,
            n_bottom=1,
            top_char="-",
            bottom_char="-",
        )
        if self.config.dataset.auto_download:
            download_dataset(
                dataset_name=self.config.dataset.name,
                dataset_dirpath=self.dataset_config.dataset_dirpath.parent,
                split=self.dataset_config.split,
                classes=self.dataset_config.classes_of_interest,
            )

        logger.info(f"Creating evaluation dataset `{self.config.dataset.name}`")
        if self.config.dataset.name == "coco":
            return COCODataset(config=self.dataset_config)

        if self.config.dataset.name == "open_images":
            return OpenImagesDataset(config=self.dataset_config)

        msg = f"Unsupported evaluation dataset `{self.config.dataset.name}`"
        logger.error(msg)
        raise ValueError(msg)

    def _evaluate_model(
        self,
        model_config: DetectionEvaluatorModelConfig,
        dataset: BaseDetectionDataset,
        num_samples: int,
        save_targets: bool,
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

        logger.info(
            f"Starting {self.config.evaluation.warmup_iterations} iterations warmup for model `{model_spec.name}`"
        )
        self._warmup_model(engine=engine, dataset=dataset)
        logger.info(f"Finished warmup for model `{model_spec.name}`")

        metric = DetectionMeanAveragePrecision(config=self.config.evaluation.metrics)
        map_progress_update_interval = self.config.evaluation.map_progress_update_interval
        latencies = []
        num_latency_images = 0
        targets_filepath = self.run_dirpath / "targets.jsonl"
        predictions_filepath = model_dirpath / "predictions.jsonl"
        latency_filepath = model_dirpath / "latency.jsonl"
        num_plotted_samples = 0
        latest_map = "unknown"

        with create_progress(unit="samples", console_width=300, bar_width=50) as progress:
            task_id = progress.add_task(
                f"Evaluating `{model_spec.name}`",
                total=num_samples,
                metrics={},
            )
            for batch_index, batch in enumerate(self._iter_batches(dataset=dataset, num_samples=num_samples)):
                sample_indices, images, targets = batch
                image_ids = [target["image_id"] for target in targets]
                if save_targets:
                    self._append_targets(
                        filepath=targets_filepath,
                        sample_indices=sample_indices,
                        targets=targets,
                    )
                prepared_images = [self._prepare_image(image=image) for image in images]
                prediction_batch = engine(images=prepared_images, image_ids=image_ids)
                predictions = list(prediction_batch.predictions)
                metric.update(predictions=predictions, targets=targets)
                if prediction_batch.latency is not None:
                    latencies.append(prediction_batch.latency)
                    num_latency_images += len(targets)
                    self._append_latency(
                        filepath=latency_filepath,
                        batch_index=batch_index,
                        sample_indices=sample_indices,
                        image_ids=image_ids,
                        latency=prediction_batch.latency,
                    )
                if self.config.outputs.save_predictions:
                    self._append_predictions(
                        filepath=predictions_filepath,
                        sample_indices=sample_indices,
                        predictions=predictions,
                    )
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

                progress_metrics = {}
                if prediction_batch.latency is not None:
                    progress_metrics["latency_ms"] = f"{prediction_batch.latency.total_ms:.2f}"
                completed_samples = progress.tasks[task_id].completed + len(targets)
                is_map_update_batch = (
                    map_progress_update_interval is not None and (batch_index + 1) % map_progress_update_interval == 0
                )
                is_last_batch = completed_samples >= num_samples
                if is_map_update_batch or is_last_batch:
                    running_metrics = metric.compute()
                    running_map = running_metrics.get("map")
                    if isinstance(running_map, int | float):
                        latest_map = f"{running_map:.4f}"
                progress_metrics["map"] = latest_map

                progress.update(
                    task_id,
                    advance=len(targets),
                    metrics=progress_metrics,
                )

        metrics = metric.compute()
        latency_summary = self._summarize_latencies(latencies=latencies, num_images=num_latency_images)
        logger.info(
            f"Metrics for model `{model_spec.name}`: \n"
            f"\tmap: {self._format_metric_value(value=metrics.get('map'))}, \n"
            f"\tmap_50: {self._format_metric_value(value=metrics.get('map_50'))}, \n"
            f"\tmap_75: {self._format_metric_value(value=metrics.get('map_75'))}"
        )
        logger.info(
            f"Average latency for model `{model_spec.name}`: \n"
            f"\tpreprocess: {latency_summary.preprocess_mean_ms:.2f} ms, \n"
            f"\tinference: {latency_summary.inference_mean_ms:.2f} ms, \n"
            f"\tpostprocess: {latency_summary.postprocess_mean_ms:.2f} ms, \n"
            f"\ttotal: {latency_summary.total_mean_ms:.2f} ms, \n"
            f"\ttotal p50: {latency_summary.total_p50_ms:.2f} ms, \n"
            f"\ttotal p95: {latency_summary.total_p95_ms:.2f} ms, \n"
            f"\ttotal p99: {latency_summary.total_p99_ms:.2f} ms, \n"
            f"\tthroughput: {latency_summary.throughput_images_per_second:.2f} images/s, \n"
            f"\tnum_images: {latency_summary.num_images}, \n"
            f"\tnum_batches: {latency_summary.num_batches}"
        )
        save_json(filepath=model_dirpath / "metrics.json", data=metrics)
        save_json(filepath=model_dirpath / "latency_summary.json", data=latency_summary.model_dump(mode="json"))

        result = DetectionEvaluationModelResult(
            model_name=model_spec.name,
            engine_name=engine.engine_name,
            metrics=metrics,
            latency=latency_summary,
            num_samples=num_samples,
            run_dirpath=model_dirpath,
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
        require_dirpath(dirpath=model_dirpath, description=f"local model `{model_spec.name}`")

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

    def _iter_batches(
        self,
        dataset: BaseDetectionDataset,
        num_samples: int,
    ) -> Iterator[tuple[list[int], list[object], list[DetectionTarget]]]:
        batch_size = self.config.evaluation.batch_size
        for start_index in range(0, num_samples, batch_size):
            sample_indices = list(range(start_index, min(start_index + batch_size, num_samples)))
            batch_items = [dataset[index] for index in sample_indices]
            images, targets = zip(*batch_items)
            yield sample_indices, list(images), list(targets)

    def _resolve_num_samples(self, dataset: BaseDetectionDataset) -> int:
        available_num_samples = len(dataset)
        requested_num_samples = self.config.evaluation.num_samples
        if requested_num_samples is None:
            return available_num_samples

        if requested_num_samples <= available_num_samples:
            return requested_num_samples

        msg = (
            f"Requested `{requested_num_samples}` evaluation samples, "
            f"but dataset contains only `{available_num_samples}` samples"
        )
        logger.error(msg)
        raise ValueError(msg)

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
        save_yaml(filepath=self.run_dirpath / "config.yaml", data=self.config.model_dump(mode="json"))

        resolved_config_data = self.config.model_dump(mode="python")
        resolved_config_data["run_name"] = self.run_name
        resolved_config_data["run_dirpath"] = self.run_dirpath
        resolved_config = DetectionEvaluatorResolvedConfig.model_validate(resolved_config_data)
        save_json(
            filepath=self.run_dirpath / "resolved_config.json",
            data=resolved_config.model_dump(mode="json"),
        )

    def _save_class_map(self, dataset: BaseDetectionDataset) -> None:
        class_ids = dataset.get_class_ids()
        class_names = dataset.get_class_names()
        source_class_ids = dataset.get_source_class_ids()
        classes = []
        for class_id, class_name, source_class_id in zip(class_ids, class_names, source_class_ids):
            class_record = DetectionClassArtifact(
                label=class_id,
                name=class_name,
                source_id=to_jsonable(value=source_class_id),
            )
            classes.append(class_record)

        class_map = DetectionClassMapArtifact(
            dataset_name=self.config.dataset.name,
            class_space=self.config.dataset.name,
            classes=tuple(classes),
        )
        save_json(filepath=self.run_dirpath / "class_map.json", data=class_map.model_dump(mode="json"))

    def _append_targets(
        self,
        filepath: Path,
        sample_indices: list[int],
        targets: list[DetectionTarget],
    ) -> None:
        records = []
        for sample_index, target in zip(sample_indices, targets):
            target_record_data = {
                "sample_index": sample_index,
                "box_format": "xyxy",
            }
            target_record_data.update(to_jsonable(value=target))
            target_record = DetectionTargetArtifact.model_validate(target_record_data)
            records.append(target_record.model_dump(mode="json"))
        append_jsonl(filepath=filepath, records=records)

    def _append_predictions(
        self,
        filepath: Path,
        sample_indices: list[int],
        predictions: list[DetectionPrediction],
    ) -> None:
        records = []
        for sample_index, prediction in zip(sample_indices, predictions):
            prediction_record_data = {
                "sample_index": sample_index,
                "box_format": "xyxy",
            }
            prediction_record_data.update(to_jsonable(value=prediction.model_dump(mode="python")))
            prediction_record = DetectionPredictionArtifact.model_validate(prediction_record_data)
            records.append(prediction_record.model_dump(mode="json"))
        append_jsonl(filepath=filepath, records=records)

    @staticmethod
    def _append_latency(
        filepath: Path,
        batch_index: int,
        sample_indices: list[int],
        image_ids: list[ImageId],
        latency: DetectionLatency,
    ) -> None:
        latency_record = DetectionLatencyArtifact(
            batch_index=batch_index,
            sample_indices=tuple(sample_indices),
            image_ids=tuple(image_ids),
            batch_size=len(sample_indices),
            preprocess_ms=latency.preprocess_ms,
            inference_ms=latency.inference_ms,
            postprocess_ms=latency.postprocess_ms,
            total_ms=latency.total_ms,
        )
        records = [latency_record.model_dump(mode="json")]
        append_jsonl(
            filepath=filepath,
            records=records,
        )

    @staticmethod
    def _format_metric_value(value: object) -> str:
        if isinstance(value, int | float):
            return f"{value:.4f}"
        return str(value)

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
    def _summarize_latencies(
        latencies: list[DetectionLatency],
        num_images: int,
    ) -> DetectionEvaluationLatencySummary:
        if not latencies:
            return DetectionEvaluationLatencySummary(
                preprocess_mean_ms=0.0,
                inference_mean_ms=0.0,
                postprocess_mean_ms=0.0,
                total_mean_ms=0.0,
                total_std_ms=0.0,
                total_min_ms=0.0,
                total_p50_ms=0.0,
                total_p90_ms=0.0,
                total_p95_ms=0.0,
                total_p99_ms=0.0,
                total_max_ms=0.0,
                throughput_images_per_second=0.0,
                num_images=0,
                num_batches=0,
            )

        preprocess_mean_ms = float(np.mean([latency.preprocess_ms for latency in latencies]))
        inference_mean_ms = float(np.mean([latency.inference_ms for latency in latencies]))
        postprocess_mean_ms = float(np.mean([latency.postprocess_ms for latency in latencies]))
        total_latencies_ms = np.asarray([latency.total_ms for latency in latencies], dtype=np.float64)
        total_mean_ms = float(np.mean(total_latencies_ms))
        total_latency_ms = float(np.sum(total_latencies_ms))
        throughput_images_per_second = num_images / (total_latency_ms / 1000) if total_latency_ms > 0 else 0.0
        return DetectionEvaluationLatencySummary(
            preprocess_mean_ms=preprocess_mean_ms,
            inference_mean_ms=inference_mean_ms,
            postprocess_mean_ms=postprocess_mean_ms,
            total_mean_ms=total_mean_ms,
            total_std_ms=float(np.std(total_latencies_ms)),
            total_min_ms=float(np.min(total_latencies_ms)),
            total_p50_ms=float(np.percentile(total_latencies_ms, 50)),
            total_p90_ms=float(np.percentile(total_latencies_ms, 90)),
            total_p95_ms=float(np.percentile(total_latencies_ms, 95)),
            total_p99_ms=float(np.percentile(total_latencies_ms, 99)),
            total_max_ms=float(np.max(total_latencies_ms)),
            throughput_images_per_second=throughput_images_per_second,
            num_images=num_images,
            num_batches=len(latencies),
        )
