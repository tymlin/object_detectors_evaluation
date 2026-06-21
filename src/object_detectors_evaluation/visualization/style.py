from matplotlib import rcParams

DEFAULT_MATPLOTLIB_FONT_FAMILY = "Arial"
DEFAULT_MATPLOTLIB_FONT_SIZE = 10.0
DEFAULT_MATPLOTLIB_FIGURE_DPI = 120
DEFAULT_MATPLOTLIB_SAVEFIG_DPI = 240

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
            "figure.dpi": DEFAULT_MATPLOTLIB_FIGURE_DPI,
            "savefig.dpi": DEFAULT_MATPLOTLIB_SAVEFIG_DPI,
        }
    )
