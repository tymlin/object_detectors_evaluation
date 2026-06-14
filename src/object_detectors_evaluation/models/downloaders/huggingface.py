from inspect import signature
from pathlib import Path

from huggingface_hub import snapshot_download

from object_detectors_evaluation.consts import MODELS_DIRPATH
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.models import DownloadedModel, ModelSpec
from object_detectors_evaluation.models.downloaders.utils import build_downloaded_model, get_model_dirpath


def download_huggingface_model(
    spec: ModelSpec,
    models_dirpath: str | Path = MODELS_DIRPATH,
    **kwargs: object,
) -> DownloadedModel:
    """Download a Hugging Face model snapshot.

    :param spec: Hugging Face-backed model spec.
    :param models_dirpath: Root directory where model artifacts are stored.
    :param kwargs: Additional arguments forwarded to ``huggingface_hub.snapshot_download``.
    :return: Downloaded model metadata.
    """
    if spec.source_type != "huggingface":
        msg = f"Model `{spec.name}` has source type `{spec.source_type}`, expected `huggingface`"
        logger.error(msg)
        raise ValueError(msg)
    if spec.repo_id is None:
        msg = f"Model `{spec.name}` has no Hugging Face repo_id"
        logger.error(msg)
        raise ValueError(msg)

    dirpath = get_model_dirpath(spec=spec, models_dirpath=models_dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Hugging Face model `{spec.name}` to path: '{dirpath}'")

    snapshot_kwargs = {"repo_id": spec.repo_id, "local_dir": dirpath}
    if "local_dir_use_symlinks" in signature(snapshot_download).parameters:
        snapshot_kwargs["local_dir_use_symlinks"] = False
    snapshot_kwargs.update(kwargs)

    snapshot_dirpath = Path(snapshot_download(**snapshot_kwargs))
    filepaths = tuple(filepath for filepath in snapshot_dirpath.rglob("*") if filepath.is_file())
    logger.info(f"Downloaded Hugging Face model `{spec.name}` to path: '{snapshot_dirpath}'")
    return build_downloaded_model(spec=spec, dirpath=snapshot_dirpath, filepaths=filepaths)
