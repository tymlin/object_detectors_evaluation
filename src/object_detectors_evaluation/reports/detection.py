from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from natsort import natsorted

from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.reports.types import LatencyField
from object_detectors_evaluation.utils import (
    iter_jsonl,
    load_json,
    load_jsonl,
    require_dirpath,
    save_json,
    save_text,
)
from object_detectors_evaluation.visualization import configure_matplotlib

METRIC_COLUMNS = (
    "map",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "mar_100",
)
LATENCY_COLUMNS = (
    "preprocess_mean_ms",
    "inference_mean_ms",
    "postprocess_mean_ms",
    "total_mean_ms",
    "total_std_ms",
    "total_min_ms",
    "total_p50_ms",
    "total_p90_ms",
    "total_p95_ms",
    "total_p99_ms",
    "total_max_ms",
    "throughput_images_per_second",
    "num_images",
    "num_batches",
)
LEADERBOARD_COLUMNS = (
    "model",
    "engine",
    "device",
    "dtype",
    "score_threshold",
    "max_detections",
    *METRIC_COLUMNS,
    "inference_mean_ms",
    "total_mean_ms",
    "total_p50_ms",
    "total_p95_ms",
    "total_p99_ms",
    "throughput_images_per_second",
)

_VIRIDIS_COLORMAP = plt.get_cmap("viridis")
_METRICS_COMPARISON_COLORS = tuple(_VIRIDIS_COLORMAP(position) for position in (0.15, 0.5, 0.85))
_OBJECT_SIZE_MAP_COLORS = ("#9ECAE1", "#4292C6", "#08519C")
_LATENCY_BREAKDOWN_COLORS = ("#6A00A8", "#CC4778", "#FCA636")


def generate_detection_evaluation_report(
    run_dirpath: str | Path,
    output_dirpath: str | Path | None = None,
    score_threshold: float = 0.25,
    top_classes: int = 20,
    model_names: list[str] | None = None,
    exclude_model_names: list[str] | None = None,
    top_models: int | None = None,
    latency_field: LatencyField = "inference_mean_ms",
    overwrite: bool = False,
) -> Path:
    """Generate a static HTML and PNG report for one completed detection evaluation run.

    :param run_dirpath: Completed detection evaluation run directory.
    :param output_dirpath: Optional output directory. Defaults to ``<run>/report``.
    :param score_threshold: Common threshold for prediction-derived report plots.
    :param top_classes: Number of ground-truth classes included in the per-class heatmap.
    :param model_names: Optional model names to include.
    :param exclude_model_names: Optional model names to exclude.
    :param top_models: Optional number of highest-mAP models to include.
    :param latency_field: Latency field used by the accuracy-versus-latency plot.
    :param overwrite: Whether an existing report directory may be replaced.
    :return: Generated report HTML filepath.
    """
    configure_matplotlib()
    _validate_report_arguments(
        score_threshold=score_threshold,
        top_classes=top_classes,
        top_models=top_models,
        latency_field=latency_field,
    )
    run_dirpath = require_dirpath(dirpath=run_dirpath, description="evaluation run")
    output_dirpath = Path(output_dirpath) if output_dirpath is not None else run_dirpath / "report"
    _validate_output_dirpath(run_dirpath=run_dirpath, output_dirpath=output_dirpath)

    logger.info(f"Generating detection evaluation report from path: '{run_dirpath}'")
    resolved_config = load_json(filepath=run_dirpath / "resolved_config.json")
    summary = load_json(filepath=run_dirpath / "summary.json")
    class_map = load_json(filepath=run_dirpath / "class_map.json")
    target_counts = _load_target_counts(filepath=run_dirpath / "targets.jsonl")

    configured_models = {model_config["name"]: model_config for model_config in resolved_config.get("models", [])}
    completed_models = {model_result["model_name"]: model_result for model_result in summary.get("models", [])}
    missing_config_models = natsorted(set(completed_models) - set(configured_models))
    if missing_config_models:
        msg = f"Could not find resolved configuration for completed models: {missing_config_models}"
        logger.error(msg)
        raise ValueError(msg)
    selected_model_names = _select_model_names(
        available_model_names=list(completed_models),
        model_names=model_names,
        exclude_model_names=exclude_model_names,
    )

    metrics_by_model: dict[str, dict[str, object]] = {}
    latency_records_by_model: dict[str, list[dict[str, object]]] = {}
    leaderboard_rows = []
    for model_name in selected_model_names:
        model_dirpath = require_dirpath(
            dirpath=run_dirpath / "models" / model_name,
            description=f"model `{model_name}`",
        )
        metrics = load_json(filepath=model_dirpath / "metrics.json")
        latency_summary = load_json(filepath=model_dirpath / "latency_summary.json")
        latency_records = load_jsonl(filepath=model_dirpath / "latency.jsonl")
        model_config = configured_models[model_name]
        model_result = completed_models[model_name]
        row = _build_leaderboard_row(
            model_name=model_name,
            model_result=model_result,
            model_config=model_config,
            metrics=metrics,
            latency_summary=latency_summary,
        )
        leaderboard_rows.append(row)
        metrics_by_model[model_name] = metrics
        latency_records_by_model[model_name] = latency_records

    leaderboard = pd.DataFrame(leaderboard_rows)
    leaderboard = leaderboard.sort_values("map", ascending=False, na_position="last").reset_index(drop=True)
    if top_models is not None:
        leaderboard = leaderboard.head(top_models).copy()
    selected_model_names = leaderboard["model"].tolist()
    metrics_by_model = {name: metrics_by_model[name] for name in selected_model_names}
    latency_records_by_model = {name: latency_records_by_model[name] for name in selected_model_names}

    warnings = _compatibility_warnings(
        leaderboard=leaderboard,
        score_threshold=score_threshold,
    )
    prediction_stats = _load_prediction_stats(
        run_dirpath=run_dirpath,
        model_names=selected_model_names,
        score_threshold=score_threshold,
        warnings=warnings,
    )

    _prepare_output_dirpath(output_dirpath=output_dirpath, overwrite=overwrite)
    figures_dirpath = output_dirpath / "figures"
    figures_dirpath.mkdir()

    leaderboard_filepath = output_dirpath / "leaderboard.csv"
    logger.info(f"Saving report leaderboard to path: '{leaderboard_filepath}'")
    leaderboard.loc[:, LEADERBOARD_COLUMNS].to_csv(leaderboard_filepath, index=False)

    figures = []
    figures.append(
        _save_metrics_comparison(
            leaderboard=leaderboard,
            filepath=figures_dirpath / "metrics_comparison.png",
        )
    )
    figures.append(
        _save_object_size_map(
            leaderboard=leaderboard,
            filepath=figures_dirpath / "object_size_map.png",
        )
    )
    figures.append(
        _save_latency_breakdown(
            leaderboard=leaderboard,
            filepath=figures_dirpath / "latency_breakdown.png",
        )
    )
    figures.append(
        _save_latency_distribution(
            latency_records_by_model=latency_records_by_model,
            filepath=figures_dirpath / "latency_distribution.png",
        )
    )
    figures.append(
        _save_accuracy_latency(
            leaderboard=leaderboard,
            latency_field=latency_field,
            filepath=figures_dirpath / "accuracy_latency.png",
        )
    )
    per_class_figure = _save_per_class_map(
        model_names=selected_model_names,
        metrics_by_model=metrics_by_model,
        class_map=class_map,
        target_counts=target_counts,
        top_classes=top_classes,
        filepath=figures_dirpath / "per_class_map.png",
    )
    if per_class_figure is not None:
        figures.append(per_class_figure)
    else:
        warnings.append("Per-class accuracy is unavailable because the run does not contain per-class metrics.")

    if prediction_stats:
        figures.append(
            _save_prediction_counts(
                prediction_stats=prediction_stats,
                filepath=figures_dirpath / "prediction_counts.png",
            )
        )
        figures.append(
            _save_confidence_distribution(
                prediction_stats=prediction_stats,
                score_threshold=score_threshold,
                filepath=figures_dirpath / "confidence_distribution.png",
            )
        )

    report_config = {
        "run_dirpath": str(run_dirpath),
        "output_dirpath": str(output_dirpath),
        "score_threshold": score_threshold,
        "top_classes": top_classes,
        "model_names": selected_model_names,
        "latency_field": latency_field,
    }
    save_json(filepath=output_dirpath / "report_config.json", data=report_config)

    leaderboard_records = json.loads(leaderboard.loc[:, LEADERBOARD_COLUMNS].to_json(orient="records"))
    report_data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": summary.get("run_name"),
        "dataset_name": summary.get("dataset_name"),
        "dataset_split": summary.get("dataset_split"),
        "num_samples": summary.get("num_samples"),
        "num_models": len(selected_model_names),
        "score_threshold": score_threshold,
        "latency_field": latency_field,
        "warnings": warnings,
        "leaderboard": leaderboard_records,
        "figures": figures,
    }
    save_json(filepath=output_dirpath / "report_data.json", data=report_data)

    html = _build_html(
        report_data=report_data,
        leaderboard=leaderboard.loc[:, LEADERBOARD_COLUMNS],
    )
    report_filepath = output_dirpath / "index.html"
    save_text(filepath=report_filepath, text=html)
    logger.info(f"Generated detection evaluation report at path: '{report_filepath}'")
    return report_filepath


def _validate_report_arguments(
    score_threshold: float,
    top_classes: int,
    top_models: int | None,
    latency_field: str,
) -> None:
    if not 0 <= score_threshold <= 1:
        msg = f"Report score threshold must be between 0 and 1, got `{score_threshold}`"
        logger.error(msg)
        raise ValueError(msg)
    if top_classes < 1:
        msg = f"Report top classes must be at least 1, got `{top_classes}`"
        logger.error(msg)
        raise ValueError(msg)
    if top_models is not None and top_models < 1:
        msg = f"Report top models must be at least 1, got `{top_models}`"
        logger.error(msg)
        raise ValueError(msg)
    if latency_field not in ("inference_mean_ms", "total_mean_ms"):
        msg = f"Unsupported report latency field `{latency_field}`"
        logger.error(msg)
        raise ValueError(msg)


def _validate_output_dirpath(run_dirpath: Path, output_dirpath: Path) -> None:
    resolved_run_dirpath = run_dirpath.resolve()
    resolved_output_dirpath = output_dirpath.resolve()
    if resolved_output_dirpath == resolved_run_dirpath or resolved_run_dirpath.is_relative_to(resolved_output_dirpath):
        msg = f"Report output directory cannot be the evaluation run or its parent: '{output_dirpath}'"
        logger.error(msg)
        raise ValueError(msg)


def _prepare_output_dirpath(output_dirpath: Path, overwrite: bool) -> None:
    if output_dirpath.exists():
        if not overwrite:
            msg = f"Report output directory already exists at path: '{output_dirpath}'"
            logger.error(msg)
            raise FileExistsError(msg)
        logger.info(f"Removing existing report output directory at path: '{output_dirpath}'")
        rmtree(output_dirpath)
    output_dirpath.mkdir(parents=True)


def _select_model_names(
    available_model_names: list[str],
    model_names: list[str] | None,
    exclude_model_names: list[str] | None,
) -> list[str]:
    available = set(available_model_names)
    requested = available_model_names if model_names is None else model_names
    unknown = natsorted(set(requested) - available)
    if unknown:
        msg = f"Report requested unknown or incomplete models: {unknown}"
        logger.error(msg)
        raise ValueError(msg)

    excluded = set(exclude_model_names or [])
    unknown_excluded = natsorted(excluded - available)
    if unknown_excluded:
        msg = f"Report excluded unknown or incomplete models: {unknown_excluded}"
        logger.error(msg)
        raise ValueError(msg)

    selected = [name for name in requested if name not in excluded]
    if not selected:
        msg = "Report model selection is empty"
        logger.error(msg)
        raise ValueError(msg)
    return selected


def _build_leaderboard_row(
    model_name: str,
    model_result: dict[str, object],
    model_config: dict[str, object],
    metrics: dict[str, object],
    latency_summary: dict[str, object],
) -> dict[str, object]:
    inference_config = model_config.get("config", {})
    row = {
        "model": model_name,
        "engine": model_result.get("engine_name"),
        "device": inference_config.get("device"),
        "dtype": inference_config.get("dtype"),
        "score_threshold": inference_config.get("score_threshold"),
        "max_detections": inference_config.get("max_detections"),
    }
    for metric_name in METRIC_COLUMNS:
        metric_value = metrics.get(metric_name)
        if isinstance(metric_value, int | float) and metric_value < 0:
            metric_value = None
        row[metric_name] = metric_value
    for latency_name in LATENCY_COLUMNS:
        row[latency_name] = latency_summary.get(latency_name)
    return row


def _compatibility_warnings(leaderboard: pd.DataFrame, score_threshold: float) -> list[str]:
    warnings = []
    for field_name in ("device", "dtype", "score_threshold", "max_detections"):
        values = leaderboard[field_name].astype(str).unique().tolist()
        if len(values) > 1:
            warnings.append(f"Models use different {field_name} values: {values}")

    unavailable_models = leaderboard.loc[
        leaderboard["score_threshold"].fillna(0).astype(float) > score_threshold,
        "model",
    ].tolist()
    if unavailable_models:
        warnings.append(
            "The report score threshold is lower than the saved prediction threshold for models: "
            f"{unavailable_models}"
        )
    return warnings


def _load_prediction_stats(
    run_dirpath: Path,
    model_names: list[str],
    score_threshold: float,
    warnings: list[str],
) -> dict[str, dict[str, object]]:
    histogram_edges = _confidence_histogram_edges(score_threshold=score_threshold)
    prediction_stats = {}
    missing_models = []
    for model_name in model_names:
        predictions_filepath = run_dirpath / "models" / model_name / "predictions.jsonl"
        if not predictions_filepath.exists():
            missing_models.append(model_name)
            continue

        counts = []
        histogram = np.zeros(len(histogram_edges) - 1, dtype=np.int64)
        for prediction in iter_jsonl(filepath=predictions_filepath):
            scores = np.asarray(prediction.get("scores", []), dtype=np.float64)
            scores = scores[scores >= score_threshold]
            counts.append(int(scores.size))
            batch_histogram, _ = np.histogram(scores, bins=histogram_edges)
            histogram += batch_histogram

        if not counts:
            missing_models.append(model_name)
            continue
        prediction_stats[model_name] = {
            "counts": counts,
            "histogram": histogram,
            "histogram_edges": histogram_edges,
        }

    if missing_models:
        warnings.append(f"Prediction-derived plots exclude models without predictions.jsonl: {missing_models}")
    return prediction_stats


def _load_target_counts(filepath: Path) -> Counter[int]:
    target_counts: Counter[int] = Counter()
    for target in iter_jsonl(filepath=filepath):
        target_counts.update(int(label) for label in target.get("labels", []))
    return target_counts


def _confidence_histogram_edges(score_threshold: float) -> np.ndarray:
    if score_threshold == 1:
        return np.asarray([1 - np.finfo(np.float64).eps, 1.0], dtype=np.float64)
    return np.linspace(score_threshold, 1.0, 31)


def _save_metrics_comparison(leaderboard: pd.DataFrame, filepath: Path) -> dict[str, str]:
    data = leaderboard.sort_values("map", ascending=True)
    model_names = data["model"].tolist()
    positions = np.arange(len(data))
    fig, axis = plt.subplots(figsize=(13, _figure_height(len(data))))
    bar_height = 0.24
    for offset, metric_name, label, color in (
        (-bar_height, "map", "mAP", _METRICS_COMPARISON_COLORS[0]),
        (0.0, "map_50", "mAP50", _METRICS_COMPARISON_COLORS[1]),
        (bar_height, "map_75", "mAP75", _METRICS_COMPARISON_COLORS[2]),
    ):
        axis.barh(positions + offset, data[metric_name], height=bar_height, label=label, color=color)
    axis.set_yticks(positions, labels=model_names)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Average precision")
    axis.set_title("Detection accuracy by model")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    _save_figure(fig=fig, filepath=filepath)
    return _figure_record("Accuracy metrics", filepath.name, "mAP, mAP50, and mAP75 by model.")


def _save_object_size_map(leaderboard: pd.DataFrame, filepath: Path) -> dict[str, str]:
    data = leaderboard.sort_values("map", ascending=True)
    model_names = data["model"].tolist()
    positions = np.arange(len(data))
    fig, axis = plt.subplots(figsize=(13, _figure_height(len(data))))
    bar_height = 0.24
    for offset, metric_name, label, color in (
        (-bar_height, "map_small", "Small", _OBJECT_SIZE_MAP_COLORS[0]),
        (0.0, "map_medium", "Medium", _OBJECT_SIZE_MAP_COLORS[1]),
        (bar_height, "map_large", "Large", _OBJECT_SIZE_MAP_COLORS[2]),
    ):
        axis.barh(positions + offset, data[metric_name], height=bar_height, label=label, color=color)
    axis.set_yticks(positions, labels=model_names)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Mean average precision")
    axis.set_title("Accuracy by object size")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    _save_figure(fig=fig, filepath=filepath)
    return _figure_record("Object-size accuracy", filepath.name, "mAP for small, medium, and large objects.")


def _save_latency_breakdown(leaderboard: pd.DataFrame, filepath: Path) -> dict[str, str]:
    data = leaderboard.sort_values("total_mean_ms", ascending=True)
    model_names = data["model"].tolist()
    fig, axis = plt.subplots(figsize=(13, _figure_height(len(data))))
    left = np.zeros(len(data), dtype=np.float64)
    for field_name, label, color in (
        ("preprocess_mean_ms", "Preprocess", _LATENCY_BREAKDOWN_COLORS[0]),
        ("inference_mean_ms", "Inference", _LATENCY_BREAKDOWN_COLORS[1]),
        ("postprocess_mean_ms", "Postprocess", _LATENCY_BREAKDOWN_COLORS[2]),
    ):
        values = data[field_name].to_numpy(dtype=np.float64)
        axis.barh(model_names, values, left=left, label=label, color=color)
        left += values
    axis.set_xlabel("Mean latency (ms per inference call)")
    axis.set_title("Latency component breakdown")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    _save_figure(fig=fig, filepath=filepath)
    return _figure_record("Latency breakdown", filepath.name, "Mean preprocessing, inference, and postprocessing.")


def _save_latency_distribution(
    latency_records_by_model: dict[str, list[dict[str, object]]],
    filepath: Path,
) -> dict[str, str]:
    model_names = []
    values = []
    for model_name, latency_records in latency_records_by_model.items():
        model_values = [float(record["total_ms"]) for record in latency_records]
        if model_values:
            model_names.append(model_name)
            values.append(model_values)

    if not values:
        msg = "Could not generate latency distribution because raw latency records are empty"
        logger.error(msg)
        raise ValueError(msg)

    fig, axis = plt.subplots(figsize=(13, _figure_height(len(model_names))))
    axis.boxplot(values, vert=False, tick_labels=model_names, showfliers=False, patch_artist=True)
    axis.set_xlabel("Total latency (ms per inference call)")
    axis.set_title("Raw latency distributions")
    axis.grid(axis="x", alpha=0.25)
    _save_figure(fig=fig, filepath=filepath)
    return _figure_record("Latency distributions", filepath.name, "Raw total latency measured for each inference call.")


def _save_accuracy_latency(
    leaderboard: pd.DataFrame,
    latency_field: LatencyField,
    filepath: Path,
) -> dict[str, str]:
    data = leaderboard.dropna(subset=["map", latency_field]).copy()
    fig, axis = plt.subplots(figsize=(12, 8))
    scatter = axis.scatter(
        data[latency_field],
        data["map"],
        c=data["map"],
        cmap=_VIRIDIS_COLORMAP,
        vmin=0,
        vmax=1,
        s=45,
    )
    fig.colorbar(scatter, ax=axis, label="mAP")
    for row in data.itertuples(index=False):
        axis.annotate(
            row.model,
            (getattr(row, latency_field), row.map),
            fontsize="x-small",
            xytext=(4, 4),
            textcoords="offset points",
        )

    pareto = _pareto_frontier(data=data, latency_field=latency_field)
    if not pareto.empty:
        axis.plot(pareto[latency_field], pareto["map"], color="#CC4778", linewidth=1.5, label="Pareto frontier")
        axis.legend()
    axis.set_xlabel(f"{latency_field} (ms per inference call)")
    axis.set_ylabel("mAP")
    axis.set_title("Accuracy versus latency")
    axis.grid(alpha=0.25)
    _save_figure(fig=fig, filepath=filepath)
    return _figure_record("Accuracy versus latency", filepath.name, "Model accuracy, latency, and Pareto frontier.")


def _pareto_frontier(data: pd.DataFrame, latency_field: LatencyField) -> pd.DataFrame:
    sorted_data = data.sort_values(latency_field, ascending=True)
    best_map = -np.inf
    rows = []
    for _, row in sorted_data.iterrows():
        if row["map"] > best_map:
            rows.append(row)
            best_map = row["map"]
    if not rows:
        return sorted_data.iloc[0:0]
    return pd.DataFrame(rows)


def _save_per_class_map(
    model_names: list[str],
    metrics_by_model: dict[str, dict[str, object]],
    class_map: dict[str, object],
    target_counts: Counter[int],
    top_classes: int,
    filepath: Path,
) -> dict[str, str] | None:
    class_name_by_label = {
        int(class_record["label"]): str(class_record["name"]) for class_record in class_map.get("classes", [])
    }
    labels = [label for label, _ in target_counts.most_common(top_classes) if label in class_name_by_label]
    if not labels:
        return None

    matrix = np.full((len(model_names), len(labels)), np.nan, dtype=np.float64)
    for model_index, model_name in enumerate(model_names):
        metrics = metrics_by_model[model_name]
        metric_classes = metrics.get("classes")
        class_values = metrics.get("map_per_class")
        if not isinstance(metric_classes, list) or not isinstance(class_values, list):
            continue
        if len(metric_classes) != len(class_values):
            msg = (
                f"Per-class metric arrays have different lengths for model `{model_name}`: "
                f"{len(metric_classes)} classes and {len(class_values)} values"
            )
            logger.error(msg)
            raise ValueError(msg)
        value_by_label = {
            int(label): float(value)
            for label, value in zip(metric_classes, class_values, strict=True)
            if float(value) >= 0
        }
        for class_index, label in enumerate(labels):
            matrix[model_index, class_index] = value_by_label.get(label, np.nan)

    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad("#d1d5db")
    fig, axis = plt.subplots(
        figsize=(max(12, len(labels) * 0.7), _figure_height(len(model_names))),
    )
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap=colormap)
    axis.set_xticks(
        np.arange(len(labels)),
        labels=[class_name_by_label[label] for label in labels],
        rotation=60,
        ha="right",
    )
    axis.set_yticks(np.arange(len(model_names)), labels=model_names)
    axis.set_title("Per-class mAP for the most frequent target classes")
    fig.colorbar(image, ax=axis, label="mAP")
    _save_figure(fig=fig, filepath=filepath)
    return _figure_record("Per-class accuracy", filepath.name, "mAP for the most frequent ground-truth classes.")


def _save_prediction_counts(
    prediction_stats: dict[str, dict[str, object]],
    filepath: Path,
) -> dict[str, str]:
    model_names = list(prediction_stats)
    values = [prediction_stats[name]["counts"] for name in model_names]
    fig, axis = plt.subplots(figsize=(13, _figure_height(len(model_names))))
    axis.boxplot(values, vert=False, tick_labels=model_names, showfliers=False, patch_artist=True)
    axis.set_xlabel("Detections per image")
    axis.set_title("Prediction counts after report score filtering")
    axis.grid(axis="x", alpha=0.25)
    _save_figure(fig=fig, filepath=filepath)
    return _figure_record("Prediction counts", filepath.name, "Distribution of retained detections per image.")


def _save_confidence_distribution(
    prediction_stats: dict[str, dict[str, object]],
    score_threshold: float,
    filepath: Path,
) -> dict[str, str]:
    model_names = list(prediction_stats)
    matrix = []
    first_stats = prediction_stats[model_names[0]]
    edges = np.asarray(first_stats["histogram_edges"], dtype=np.float64)
    for model_name in model_names:
        stats = prediction_stats[model_name]
        histogram = np.asarray(stats["histogram"], dtype=np.float64)
        histogram_sum = histogram.sum()
        matrix.append(histogram / histogram_sum if histogram_sum > 0 else histogram)

    fig, axis = plt.subplots(figsize=(13, _figure_height(len(model_names))))
    image = axis.imshow(np.asarray(matrix), aspect="auto", cmap="magma")
    tick_indices = np.linspace(0, len(edges) - 2, min(6, len(edges) - 1), dtype=int)
    tick_labels = [f"{edges[index]:.2f}" for index in tick_indices]
    axis.set_xticks(tick_indices, labels=tick_labels)
    axis.set_yticks(np.arange(len(model_names)), labels=model_names)
    axis.set_xlabel("Confidence score")
    axis.set_title(f"Normalized confidence distributions (score >= {score_threshold:.3f})")
    fig.colorbar(image, ax=axis, label="Share of retained predictions")
    _save_figure(fig=fig, filepath=filepath)
    return _figure_record("Confidence distributions", filepath.name, "Normalized confidence histograms by model.")


def _save_figure(fig: Figure, filepath: Path) -> None:
    logger.info(f"Saving report figure to path: '{filepath}'")
    fig.tight_layout()
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)


def _figure_height(num_items: int) -> float:
    return max(5.0, min(30.0, 1.5 + num_items * 0.35))


def _figure_record(title: str, filename: str, description: str) -> dict[str, str]:
    return {
        "title": title,
        "filename": f"figures/{filename}",
        "description": description,
    }


def _build_html(report_data: dict[str, object], leaderboard: pd.DataFrame) -> str:
    warnings = report_data["warnings"]
    warning_html = "".join(f"<li>{escape(str(warning))}</li>" for warning in warnings)
    if not warning_html:
        warning_html = "<li>No comparison warnings.</li>"

    figures_html = "".join(
        (
            "<section>"
            f"<h2>{escape(figure['title'])}</h2>"
            f"<p>{escape(figure['description'])}</p>"
            f"<a href=\"{escape(figure['filename'])}\">"
            f"<img src=\"{escape(figure['filename'])}\" alt=\"{escape(figure['title'])}\"></a>"
            "</section>"
        )
        for figure in report_data["figures"]
    )
    leaderboard_html = leaderboard.to_html(
        index=False,
        classes="leaderboard",
        border=0,
        float_format=lambda value: f"{value:.4f}",
        table_id="leaderboard",
    )
    run_name = escape(str(report_data.get("run_name")))
    dataset = escape(f"{report_data.get('dataset_name')} / {report_data.get('dataset_split')}")
    generated_at = escape(str(report_data.get("generated_at")))
    num_samples = escape(str(report_data.get("num_samples")))
    num_models = escape(str(report_data.get("num_models")))
    score_threshold = escape(str(report_data.get("score_threshold")))
    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Detection evaluation report - {run_name}</title>
    <style>
        body {{ margin: 0; color: #17202a; background: #f5f7fa; font-family: Arial, sans-serif; }}
        main {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
        header {{ padding: 20px 0; border-bottom: 2px solid #17202a; }}
        h1, h2 {{ letter-spacing: 0; }}
        .meta {{ display: flex; flex-wrap: wrap; gap: 24px; color: #4b5563; }}
        section {{ margin: 28px 0; padding: 20px; background: white; border: 1px solid #d9dee7; border-radius: 6px; }}
        img {{ display: block; width: 100%; height: auto; }}
        .table-wrap {{ overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ position: sticky; top: 0; cursor: pointer; background: #e8edf4; }}
        th, td {{ padding: 8px 10px; border-bottom: 1px solid #d9dee7; text-align: right; white-space: nowrap; }}
        th:first-child, td:first-child {{ text-align: left; }}
        tr:hover {{ background: #f3f6fa; }}
        .warnings {{ border-left: 5px solid #ca8a04; }}
        a {{ color: #1d4ed8; }}
    </style>
</head>
<body>
<main>
    <header>
        <h1>Detection evaluation report</h1>
        <div class="meta">
            <span><strong>Run:</strong> {run_name}</span>
            <span><strong>Dataset:</strong> {dataset}</span>
            <span><strong>Samples:</strong> {num_samples}</span>
            <span><strong>Models:</strong> {num_models}</span>
            <span><strong>Prediction plot threshold:</strong> {score_threshold}</span>
            <span><strong>Generated:</strong> {generated_at}</span>
        </div>
    </header>
    <section class="warnings">
        <h2>Comparison notes</h2>
        <ul>{warning_html}</ul>
    </section>
    <section>
        <h2>Leaderboard</h2>
        <p>Metrics are the saved evaluator outputs. The report threshold only filters prediction-derived plots.</p>
        <p>Click a column header to sort. Numeric columns sort numerically.</p>
        <div class="table-wrap">{leaderboard_html}</div>
        <p><a href="leaderboard.csv">Download leaderboard.csv</a> |
           <a href="report_data.json">Open report_data.json</a></p>
    </section>
    {figures_html}
</main>
<script>
document.querySelectorAll('#leaderboard th').forEach((header, column) => {{
    header.addEventListener('click', () => {{
        const table = header.closest('table');
        const body = table.querySelector('tbody');
        const rows = Array.from(body.querySelectorAll('tr'));
        const ascending = header.dataset.order !== 'asc';
        rows.sort((left, right) => {{
            const a = left.children[column].textContent.trim();
            const b = right.children[column].textContent.trim();
            const aNumber = Number(a);
            const bNumber = Number(b);
            const comparison = Number.isNaN(aNumber) || Number.isNaN(bNumber)
                ? a.localeCompare(b, undefined, {{ numeric: true, sensitivity: 'base' }})
                : aNumber - bNumber;
            return ascending ? comparison : -comparison;
        }});
        rows.forEach(row => body.appendChild(row));
        header.dataset.order = ascending ? 'asc' : 'desc';
    }});
}});
</script>
</body>
</html>
"""
