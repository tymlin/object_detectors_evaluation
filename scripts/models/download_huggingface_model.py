from object_detectors_evaluation.consts import MODELS_DIRPATH
from object_detectors_evaluation.loggers import logger
from object_detectors_evaluation.models.collections import RTDETR_MODELS
from object_detectors_evaluation.models.downloaders import download_huggingface_model

MODEL_SPEC = RTDETR_MODELS[0]


def main() -> None:
    logger.info(f"Downloading Hugging Face model `{MODEL_SPEC.name}` to path: '{MODELS_DIRPATH}'")
    downloaded_model = download_huggingface_model(
        spec=MODEL_SPEC,
        models_dirpath=MODELS_DIRPATH,
    )
    logger.info(
        f"Downloaded model `{downloaded_model.spec.name}` with {len(downloaded_model.filepaths)} files "
        f"to path: '{downloaded_model.dirpath}'"
    )


if __name__ == "__main__":
    main()
