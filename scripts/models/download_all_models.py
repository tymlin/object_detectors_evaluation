from object_detectors_evaluation.consts import MODELS_DIRPATH
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.models import ModelSpec
from object_detectors_evaluation.models.downloaders import download_model
from object_detectors_evaluation.models.registry import DETECTION_MODELS

MODEL_SPECS: tuple[ModelSpec, ...] = DETECTION_MODELS
OVERWRITE = False


def main() -> None:
    logger.info(f"Downloading {len(MODEL_SPECS)} detection models to path: '{MODELS_DIRPATH}'")

    for idx, model_spec in enumerate(MODEL_SPECS, start=1):
        logger.info(f"Downloading model {idx}/{len(MODEL_SPECS)}: `{model_spec.name}`")
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
