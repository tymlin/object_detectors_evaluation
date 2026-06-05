from pathlib import Path
from urllib.request import urlretrieve

from object_detectors_evaluation.consts import MODELS_DIRPATH
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.models import DownloadedModel, ModelSpec
from object_detectors_evaluation.models.downloaders.base import build_downloaded_model, get_model_dirpath


def download_url_model(
    spec: ModelSpec,
    models_dirpath: str | Path = MODELS_DIRPATH,
    overwrite: bool = False,
) -> DownloadedModel:
    """Download a model artifact from a direct URL.

    :param spec: Direct-URL-backed model spec.
    :param models_dirpath: Root directory where model artifacts are stored.
    :param overwrite: Whether to overwrite an existing local file.
    :return: Downloaded model metadata.
    """
    if spec.source_type != "url":
        msg = f"Model `{spec.name}` has source type `{spec.source_type}`, expected `url`"
        logger.error(msg)
        raise ValueError(msg)
    if spec.model_url is None:
        msg = f"Model `{spec.name}` has no model_url"
        logger.error(msg)
        raise ValueError(msg)

    filename = spec.resolved_filename
    if filename is None:
        msg = f"Could not resolve filename for model `{spec.name}`"
        logger.error(msg)
        raise ValueError(msg)

    dirpath = get_model_dirpath(spec=spec, models_dirpath=models_dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    filepath = dirpath / filename

    if filepath.exists() and not overwrite:
        logger.info(f"Model `{spec.name}` already exists at path: '{filepath}'")
        return build_downloaded_model(spec=spec, dirpath=dirpath, filepaths=(filepath,))

    logger.info(f"Downloading model `{spec.name}` from `{spec.model_url}` to path: '{filepath}'")
    urlretrieve(spec.model_url, filepath)
    logger.info(f"Downloaded model `{spec.name}` to path: '{filepath}'")
    return build_downloaded_model(spec=spec, dirpath=dirpath, filepaths=(filepath,))
