from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIRPATH = ROOT / "configs"
DATASETS_DIRPATH = ROOT / "datasets"
FIFTYONE_DATASETS_DIRPATH = DATASETS_DIRPATH / "fiftyone"
MODELS_DIRPATH = ROOT / "models"
RUNS_DIRPATH = ROOT / "runs"
RESULTS_DIRPATH = ROOT / "results"

NOW = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

LOGGER_LEVEL = "INFO"
LOGGER_FILENAME: str | None = None
LOGGER_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<magenta>{process.name}:{thread.name}</magenta> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)
