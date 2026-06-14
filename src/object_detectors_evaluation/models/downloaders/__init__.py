from object_detectors_evaluation.models import DownloadedModel, ModelSpec
from object_detectors_evaluation.models.downloaders.base import download_model
from object_detectors_evaluation.models.downloaders.huggingface import download_huggingface_model
from object_detectors_evaluation.models.downloaders.url import download_url_model
from object_detectors_evaluation.models.downloaders.utils import build_downloaded_model, get_model_dirpath

__all__ = [
    "DownloadedModel",
    "ModelSpec",
    "build_downloaded_model",
    "download_huggingface_model",
    "download_model",
    "download_url_model",
    "get_model_dirpath",
]
