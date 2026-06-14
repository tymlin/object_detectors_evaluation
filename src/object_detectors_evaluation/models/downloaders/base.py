from pathlib import Path

from object_detectors_evaluation.consts import MODELS_DIRPATH
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.models import DownloadedModel, ModelSpec
from object_detectors_evaluation.models.downloaders.huggingface import download_huggingface_model
from object_detectors_evaluation.models.downloaders.url import download_url_model


def download_model(
    spec: ModelSpec,
    models_dirpath: str | Path = MODELS_DIRPATH,
    **kwargs: object,
) -> DownloadedModel:
    """Download a model using the downloader matching its source type.

    :param spec: Model metadata.
    :param models_dirpath: Root directory where model artifacts are stored.
    :param kwargs: Additional downloader-specific keyword arguments.
    :return: Downloaded model metadata.
    """
    if spec.source_type == "huggingface":
        return download_huggingface_model(spec=spec, models_dirpath=models_dirpath, **kwargs)
    if spec.source_type == "url":
        return download_url_model(spec=spec, models_dirpath=models_dirpath, **kwargs)

    msg = f"Model source type `{spec.source_type}` is not supported by download_model"
    logger.error(msg)
    raise ValueError(msg)
