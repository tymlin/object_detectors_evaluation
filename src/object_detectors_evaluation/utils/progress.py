from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Column
from rich.text import Text

ProgressMetrics = dict[str, object]


class CountColumn(ProgressColumn):
    """Render task progress as a count with an optional total."""

    def __init__(self, unit: str = "items", no_wrap: bool = True) -> None:
        self.unit = unit
        self.no_wrap = no_wrap
        super().__init__()

    def render(self, task: Task) -> Text:
        completed = int(task.completed)
        if task.total is None:
            return Text(f"{completed} {self.unit}")
        return Text(f"{completed}/{int(task.total)} {self.unit}")

    def get_table_column(self) -> Column:
        return Column(no_wrap=self.no_wrap)


class MetricsColumn(ProgressColumn):
    """Render tqdm-style postfix metrics from ``task.fields['metrics']``."""

    def __init__(self, no_wrap: bool = True) -> None:
        self.no_wrap = no_wrap
        super().__init__()

    def render(self, task: Task) -> Text:
        metrics = task.fields.get("metrics")
        if not metrics:
            return Text("")

        return Text(
            " ".join(f"{key}={value}" for key, value in metrics.items()),
            no_wrap=self.no_wrap,
            overflow="ellipsis",
        )

    def get_table_column(self) -> Column:
        return Column(no_wrap=self.no_wrap, overflow="ellipsis")


def normalize_progress_total(total: int | float | None) -> int | None:
    """Return a Rich-compatible total, treating non-positive values as unknown."""
    if total is None:
        return None

    total_int = int(total)
    if total_int <= 0:
        return None
    return total_int


def create_progress(
    unit: str = "items",
    show_metrics: bool = True,
    bar_width: int | None = 30,
    no_wrap: bool = True,
    console_width: int | None = None,
) -> Progress:
    """Create the standard terminal progress bar used by scripts."""
    bar_column = (
        BarColumn(bar_width=None)
        if bar_width is None
        else BarColumn(
            bar_width=bar_width,
            table_column=Column(width=bar_width, no_wrap=True),
        )
    )

    columns = [
        SpinnerColumn(),
        TextColumn(
            "[progress.description]{task.description}",
            table_column=Column(no_wrap=no_wrap, overflow="ellipsis"),
        ),
        bar_column,
        CountColumn(unit=unit, no_wrap=no_wrap),
    ]
    if show_metrics:
        columns.append(MetricsColumn(no_wrap=no_wrap))
    columns.append(TimeElapsedColumn())
    columns.append(TimeRemainingColumn())

    return Progress(
        *columns,
        console=Console(force_terminal=True, force_interactive=True, width=console_width),
        refresh_per_second=10,
    )
