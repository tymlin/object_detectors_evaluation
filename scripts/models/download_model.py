from object_detectors_evaluation.consts import MODELS_DIRPATH
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.models.downloaders import download_model
from object_detectors_evaluation.models.registry import get_detection_model_spec

MODEL_NAME = "yolov8n"
OVERWRITE = False


def main() -> None:
    model_spec = get_detection_model_spec(MODEL_NAME)
    logger.info(f"Downloading model `{model_spec.name}` to path: '{MODELS_DIRPATH}'")
    download_kwargs = {"overwrite": OVERWRITE} if model_spec.source_type == "url" else {}
    downloaded_model = download_model(
        spec=model_spec,
        models_dirpath=MODELS_DIRPATH,
        **download_kwargs,
    )
    logger.info(
        f"Downloaded model `{downloaded_model.spec.name}` with {len(downloaded_model.filepaths)} files "
        f"to path: '{downloaded_model.dirpath}'"
    )


if __name__ == "__main__":
    main()
