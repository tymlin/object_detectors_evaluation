from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colormaps
from matplotlib.colors import to_hex
from natsort import natsorted
from plotly import graph_objects as go
from plotly.io import to_html
from plotly.offline import get_plotlyjs

from object_detectors_evaluation.reports.types import LatencyField
from object_detectors_evaluation.utils import save_text

_METRICS_COMPARISON_COLORS = tuple(to_hex(colormaps["viridis"](position)) for position in (0.15, 0.5, 0.85))
_OBJECT_SIZE_MAP_COLORS = ("#9ECAE1", "#4292C6", "#08519C")
_LATENCY_BREAKDOWN_COLORS = ("#6A00A8", "#CC4778", "#FCA636")
_PARETO_COLOR = "#CC4778"

_PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "toImageButtonOptions": {
        "format": "png",
        "scale": 2,
    },
}


def save_plotly_javascript(filepath: Path) -> None:
    """Save the installed Plotly browser runtime.

    :param filepath: Output JavaScript filepath.
    """
    save_text(filepath=filepath, text=get_plotlyjs())


def build_detection_interactive_plots(
    leaderboard: pd.DataFrame,
    latency_records_by_model: dict[str, list[dict[str, object]]],
    metrics_by_model: dict[str, dict[str, object]],
    class_map: dict[str, object],
    target_counts: Counter[int],
    prediction_stats: dict[str, dict[str, object]],
    score_threshold: float,
    top_classes: int,
    latency_field: LatencyField,
) -> list[dict[str, str]]:
    """Build interactive Plotly charts for a detection report.

    :param leaderboard: Normalized model leaderboard.
    :param latency_records_by_model: Raw measured latency records by model.
    :param metrics_by_model: Evaluator metric outputs by model.
    :param class_map: Dataset class map artifact.
    :param target_counts: Ground-truth object counts by class label.
    :param prediction_stats: Streamed prediction statistics by model.
    :param score_threshold: Report score threshold.
    :param top_classes: Maximum classes shown in the per-class heatmap.
    :param latency_field: Latency value shown in the accuracy-latency chart.
    :return: Interactive plot metadata and HTML fragments.
    """
    model_names = natsorted(leaderboard["model"].tolist())
    plots = [
        _metrics_comparison_plot(leaderboard=leaderboard),
        _object_size_plot(leaderboard=leaderboard),
        _latency_breakdown_plot(leaderboard=leaderboard),
        _latency_distribution_plot(
            model_names=model_names,
            latency_records_by_model=latency_records_by_model,
        ),
        _accuracy_latency_plot(leaderboard=leaderboard, latency_field=latency_field),
    ]

    per_class_plot = _per_class_plot(
        model_names=model_names,
        metrics_by_model=metrics_by_model,
        class_map=class_map,
        target_counts=target_counts,
        top_classes=top_classes,
    )
    if per_class_plot is not None:
        plots.append(per_class_plot)

    if prediction_stats:
        plots.append(_prediction_counts_plot(prediction_stats=prediction_stats))
        plots.append(
            _confidence_distribution_plot(
                prediction_stats=prediction_stats,
                score_threshold=score_threshold,
            )
        )
    return plots


def _metrics_comparison_plot(leaderboard: pd.DataFrame) -> dict[str, str]:
    figure = go.Figure()
    model_names = leaderboard["model"].tolist()
    for metric_name, label, color in zip(
        ("map", "map_50", "map_75"),
        ("mAP", "mAP50", "mAP75"),
        _METRICS_COMPARISON_COLORS,
        strict=True,
    ):
        figure.add_bar(
            x=leaderboard[metric_name],
            y=model_names,
            name=label,
            orientation="h",
            marker_color=color,
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.4f}}<extra></extra>",
        )
    figure.update_layout(barmode="group")
    _style_figure(figure=figure, model_names=model_names, x_title="Average precision")
    _add_model_sort_menu(
        figure=figure,
        leaderboard=leaderboard,
        sort_fields=(
            ("map", "Overall mAP"),
            ("map_50", "mAP50"),
            ("map_75", "mAP75"),
            (None, "Model name"),
        ),
    )
    return _plot_record(
        plot_id="metrics-comparison",
        title="Accuracy metrics",
        description="mAP, mAP50, and mAP75 by model.",
        figure=figure,
    )


def _object_size_plot(leaderboard: pd.DataFrame) -> dict[str, str]:
    figure = go.Figure()
    model_names = leaderboard["model"].tolist()
    for metric_name, label, color in zip(
        ("map_small", "map_medium", "map_large"),
        ("Small", "Medium", "Large"),
        _OBJECT_SIZE_MAP_COLORS,
        strict=True,
    ):
        figure.add_bar(
            x=leaderboard[metric_name],
            y=model_names,
            name=label,
            orientation="h",
            marker_color=color,
            hovertemplate=f"<b>%{{y}}</b><br>{label} mAP: %{{x:.4f}}<extra></extra>",
        )
    figure.update_layout(barmode="group")
    _style_figure(figure=figure, model_names=model_names, x_title="Mean average precision")
    _add_model_sort_menu(
        figure=figure,
        leaderboard=leaderboard,
        sort_fields=(
            ("map", "Overall mAP"),
            ("map_small", "Small-object mAP"),
            ("map_medium", "Medium-object mAP"),
            ("map_large", "Large-object mAP"),
            (None, "Model name"),
        ),
    )
    return _plot_record(
        plot_id="object-size-map",
        title="Object-size accuracy",
        description="mAP for small, medium, and large objects.",
        figure=figure,
    )


def _latency_breakdown_plot(leaderboard: pd.DataFrame) -> dict[str, str]:
    figure = go.Figure()
    model_names = leaderboard["model"].tolist()
    for field_name, label, color in zip(
        ("preprocess_mean_ms", "inference_mean_ms", "postprocess_mean_ms"),
        ("Preprocess", "Inference", "Postprocess"),
        _LATENCY_BREAKDOWN_COLORS,
        strict=True,
    ):
        figure.add_bar(
            x=leaderboard[field_name],
            y=model_names,
            name=label,
            orientation="h",
            marker_color=color,
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.2f}} ms<extra></extra>",
        )
    figure.update_layout(barmode="stack", legend_traceorder="normal")
    _style_figure(figure=figure, model_names=model_names, x_title="Mean latency (ms per inference call)")
    _add_model_sort_menu(
        figure=figure,
        leaderboard=leaderboard,
        sort_fields=(
            ("total_mean_ms", "Total latency"),
            ("preprocess_mean_ms", "Preprocess latency"),
            ("inference_mean_ms", "Inference latency"),
            ("postprocess_mean_ms", "Postprocess latency"),
            (None, "Model name"),
        ),
    )
    return _plot_record(
        plot_id="latency-breakdown",
        title="Latency breakdown",
        description="Mean preprocessing, inference, and postprocessing latency.",
        figure=figure,
    )


def _latency_distribution_plot(
    model_names: list[str],
    latency_records_by_model: dict[str, list[dict[str, object]]],
) -> dict[str, str]:
    values_by_model = {
        model_name: [float(record["total_ms"]) for record in latency_records_by_model[model_name]]
        for model_name in model_names
    }
    distribution_model_names = [model_name for model_name in model_names if values_by_model[model_name]]
    figure = _distribution_figure(model_names=distribution_model_names, values_by_model=values_by_model)
    _style_figure(
        figure=figure,
        model_names=distribution_model_names,
        x_title="Total latency (ms per inference call)",
    )
    return _plot_record(
        plot_id="latency-distribution",
        title="Latency distributions",
        description="Raw total latency measured for each inference call.",
        figure=figure,
    )


def _accuracy_latency_plot(leaderboard: pd.DataFrame, latency_field: LatencyField) -> dict[str, str]:
    data = leaderboard.dropna(subset=["map", latency_field]).copy()
    customdata = data[["engine", "device", "dtype", "score_threshold", "max_detections"]].to_numpy()
    figure = go.Figure()
    figure.add_scatter(
        x=data[latency_field],
        y=data["map"],
        mode="markers+text",
        name="Models",
        text=data["model"],
        textposition="top center",
        customdata=customdata,
        marker={
            "color": data["map"],
            "colorscale": "Viridis",
            "cmin": 0,
            "cmax": 1,
            "colorbar": {"title": "mAP"},
            "size": 10,
        },
        hovertemplate=(
            "<b>%{text}</b><br>"
            "mAP: %{y:.4f}<br>"
            "Latency: %{x:.2f} ms<br>"
            "Engine: %{customdata[0]}<br>"
            "Device: %{customdata[1]}<br>"
            "Dtype: %{customdata[2]}<br>"
            "Score threshold: %{customdata[3]}<br>"
            "Max detections: %{customdata[4]}<extra></extra>"
        ),
    )

    pareto = _pareto_frontier(data=data, latency_field=latency_field)
    if not pareto.empty:
        figure.add_scatter(
            x=pareto[latency_field],
            y=pareto["map"],
            mode="lines",
            name="Pareto frontier",
            line={"color": _PARETO_COLOR, "width": 2},
            hovertemplate="Pareto frontier<extra></extra>",
        )

    latency_label = "Inference latency" if latency_field == "inference_mean_ms" else "Total pipeline latency"
    _style_figure(figure=figure, x_title=f"{latency_label} (ms per inference call)", y_title="mAP", height=560)
    return _plot_record(
        plot_id="accuracy-latency",
        title="Accuracy versus latency",
        description="Model accuracy, latency, and Pareto frontier.",
        figure=figure,
    )


def _per_class_plot(
    model_names: list[str],
    metrics_by_model: dict[str, dict[str, object]],
    class_map: dict[str, object],
    target_counts: Counter[int],
    top_classes: int,
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
        value_by_label = {
            int(label): float(value)
            for label, value in zip(metric_classes, class_values, strict=True)
            if float(value) >= 0
        }
        for class_index, label in enumerate(labels):
            matrix[model_index, class_index] = value_by_label.get(label, np.nan)

    class_names = [class_name_by_label[label] for label in labels]
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=class_names,
            y=model_names,
            zmin=0,
            zmax=1,
            colorscale="Viridis",
            colorbar={"title": "mAP"},
            hovertemplate="<b>%{y}</b><br>Class: %{x}<br>mAP: %{z:.4f}<extra></extra>",
        )
    )
    _style_figure(figure=figure, model_names=model_names, height=_plot_height(len(model_names)))
    return _plot_record(
        plot_id="per-class-map",
        title="Per-class accuracy",
        description="mAP for the most frequent ground-truth classes.",
        figure=figure,
    )


def _prediction_counts_plot(prediction_stats: dict[str, dict[str, object]]) -> dict[str, str]:
    model_names = natsorted(prediction_stats)
    values_by_model = {
        model_name: [float(value) for value in prediction_stats[model_name]["counts"]] for model_name in model_names
    }
    figure = _distribution_figure(model_names=model_names, values_by_model=values_by_model)
    _style_figure(figure=figure, model_names=model_names, x_title="Detections per image")
    return _plot_record(
        plot_id="prediction-counts",
        title="Prediction counts",
        description="Distribution of retained detections per image.",
        figure=figure,
    )


def _confidence_distribution_plot(
    prediction_stats: dict[str, dict[str, object]],
    score_threshold: float,
) -> dict[str, str]:
    model_names = natsorted(prediction_stats)
    first_stats = prediction_stats[model_names[0]]
    edges = np.asarray(first_stats["histogram_edges"], dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    matrix = []
    for model_name in model_names:
        histogram = np.asarray(prediction_stats[model_name]["histogram"], dtype=np.float64)
        histogram_sum = histogram.sum()
        matrix.append(histogram / histogram_sum if histogram_sum > 0 else histogram)

    figure = go.Figure(
        data=go.Heatmap(
            z=np.asarray(matrix),
            x=centers,
            y=model_names,
            colorscale="Magma",
            colorbar={"title": "Share"},
            hovertemplate="<b>%{y}</b><br>Confidence: %{x:.3f}<br>Share: %{z:.4f}<extra></extra>",
        )
    )
    _style_figure(
        figure=figure,
        model_names=model_names,
        x_title=f"Confidence score (threshold >= {score_threshold:.3f})",
    )
    return _plot_record(
        plot_id="confidence-distribution",
        title="Confidence distributions",
        description="Normalized confidence histograms by model.",
        figure=figure,
    )


def _distribution_figure(
    model_names: list[str],
    values_by_model: dict[str, list[float]],
) -> go.Figure:
    figure = go.Figure()
    colors = [to_hex(color) for color in sns.husl_palette(n_colors=len(model_names))]
    for model_name, color in zip(model_names, colors, strict=True):
        values = values_by_model[model_name]
        if len(values) >= 3 and len(set(values)) >= 2:
            figure.add_trace(
                go.Violin(
                    x=values,
                    y=[model_name] * len(values),
                    name=model_name,
                    orientation="h",
                    fillcolor=color,
                    line_color=color,
                    box_visible=True,
                    meanline_visible=False,
                    points=False,
                    spanmode="hard",
                    scalemode="width",
                    showlegend=False,
                    hovertemplate=f"<b>{model_name}</b><br>Value: %{{x:.3f}}<extra></extra>",
                )
            )
        else:
            figure.add_scatter(
                x=[float(np.median(values))],
                y=[model_name],
                mode="markers",
                marker={"color": color, "size": 9, "line": {"color": "black", "width": 1}},
                showlegend=False,
                hovertemplate=f"<b>{model_name}</b><br>Median: %{{x:.3f}}<extra></extra>",
            )
    return figure


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


def _style_figure(
    figure: go.Figure,
    model_names: list[str] | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    height: int | None = None,
) -> None:
    figure.update_layout(
        template="plotly_white",
        height=height or _plot_height(len(model_names or [])),
        margin={"l": 120, "r": 40, "t": 30, "b": 70},
        font={"family": "Arial, sans-serif", "size": 12, "color": "#17202a"},
        hoverlabel={"font": {"family": "Arial, sans-serif", "size": 12}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    figure.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#9ca3af")
    figure.update_yaxes(gridcolor="#e5e7eb", zeroline=False)
    if model_names is not None:
        figure.update_yaxes(categoryorder="array", categoryarray=list(reversed(model_names)))


def _plot_height(num_models: int) -> int:
    return max(420, min(1200, 220 + num_models * 34))


def _add_model_sort_menu(
    figure: go.Figure,
    leaderboard: pd.DataFrame,
    sort_fields: tuple[tuple[str | None, str], ...],
) -> None:
    buttons = []
    initial_categoryarray = []
    for field_name, label in sort_fields:
        if field_name is None:
            categoryarray = list(reversed(natsorted(leaderboard["model"].tolist())))
        else:
            sorted_leaderboard = leaderboard.sort_values(
                field_name,
                ascending=True,
                na_position="first",
                kind="stable",
            )
            categoryarray = sorted_leaderboard["model"].tolist()
        if not initial_categoryarray:
            initial_categoryarray = categoryarray
        buttons.append(
            {
                "label": f"Sort by: {label}",
                "method": "relayout",
                "args": [
                    {
                        "yaxis.categoryorder": "array",
                        "yaxis.categoryarray": categoryarray,
                    }
                ],
            }
        )

    figure.update_layout(
        updatemenus=[
            {
                "type": "dropdown",
                "direction": "down",
                "showactive": True,
                "active": 0,
                "x": 0,
                "xanchor": "left",
                "y": 1.08,
                "yanchor": "top",
                "buttons": buttons,
            }
        ],
    )
    figure.update_yaxes(categoryorder="array", categoryarray=initial_categoryarray)


def _plot_record(plot_id: str, title: str, description: str, figure: go.Figure) -> dict[str, str]:
    html = to_html(
        figure,
        config=_PLOTLY_CONFIG,
        full_html=False,
        include_plotlyjs=False,
        div_id=plot_id,
    )
    return {
        "id": plot_id,
        "title": title,
        "description": description,
        "html": html,
    }
