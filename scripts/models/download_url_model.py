from object_detectors_evaluation.consts import MODELS_DIRPATH
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.models.collections import ULTRALYTICS_YOLO_DETECTION_MODELS
from object_detectors_evaluation.models.downloaders import download_url_model

MODEL_SPEC = ULTRALYTICS_YOLO_DETECTION_MODELS[0]
OVERWRITE = False


def main() -> None:
    logger.info(f"Downloading URL model `{MODEL_SPEC.name}` to path: '{MODELS_DIRPATH}'")
    downloaded_model = download_url_model(
        spec=MODEL_SPEC,
        models_dirpath=MODELS_DIRPATH,
        overwrite=OVERWRITE,
    )
    logger.info(
        f"Downloaded model `{downloaded_model.spec.name}` with {len(downloaded_model.filepaths)} files "
        f"to path: '{downloaded_model.dirpath}'"
    )


if __name__ == "__main__":
    main()
