from matplotlib import rcParams

DEFAULT_MATPLOTLIB_FONT_FAMILY = "Arial"
DEFAULT_MATPLOTLIB_FONT_SIZE = 10.0

_MATPLOTLIB_FONT_FALLBACKS = ("Liberation Sans", "DejaVu Sans")


def configure_matplotlib() -> None:
    """Apply shared Matplotlib font defaults."""
    rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [DEFAULT_MATPLOTLIB_FONT_FAMILY, *_MATPLOTLIB_FONT_FALLBACKS],
            "font.size": DEFAULT_MATPLOTLIB_FONT_SIZE,
            "axes.titlesize": "large",
            "axes.labelsize": "medium",
            "xtick.labelsize": "small",
            "ytick.labelsize": "small",
            "legend.fontsize": "medium",
            "figure.titlesize": "large",
        }
    )
