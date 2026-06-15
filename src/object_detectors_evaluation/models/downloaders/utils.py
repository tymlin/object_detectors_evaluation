from pathlib import Path

from object_detectors_evaluation.consts import MODELS_DIRPATH
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.models import DownloadedModel, ModelSpec


def get_model_dirpath(spec: ModelSpec, models_dirpath: str | Path = MODELS_DIRPATH) -> Path:
    """Resolve the local directory for a model spec.

    :param spec: Model metadata.
    :param models_dirpath: Root directory where model artifacts are stored.
    :return: Local model directory path.
    """
    return Path(models_dirpath) / spec.name


def build_downloaded_model(spec: ModelSpec, dirpath: Path, filepaths: tuple[Path, ...]) -> DownloadedModel:
    """Create a downloaded model record.

    :param spec: Source model metadata.
    :param dirpath: Local model directory.
    :param filepaths: Local model files.
    :return: Downloaded model metadata.
    """
    logger.info(f"Built model artifact for `{spec.name}` with `{len(filepaths)}` files at path: '{dirpath}'")
    return DownloadedModel(spec=spec, dirpath=dirpath, filepaths=filepaths)
