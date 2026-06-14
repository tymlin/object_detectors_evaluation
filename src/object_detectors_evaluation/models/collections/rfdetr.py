"""Structured Hugging Face metadata scraped from the Roboflow models page.

Detection-only RF-DETR checkpoints.

Source page:
https://huggingface.co/Roboflow/models
"""

from object_detectors_evaluation.models import ModelSpec

RFDETR_MODELS_PAGE_URL = "https://huggingface.co/Roboflow/models"

_BASE_URL = "https://huggingface.co"

RFDETR_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        source_type="huggingface",
        name="rf-detr-nano",
        size="nano",
        repo_id="Roboflow/rf-detr-nano",
        model_url=f"{_BASE_URL}/Roboflow/rf-detr-nano",
        family="RF-DETR",
        training_track="detection",
        class_space="coco",
        training_dataset="COCO",
        checkpoint_format="safetensors",
        inference_engine="transformers",
        note="Detection checkpoint from the official Roboflow Hugging Face organization page.",
    ),
    ModelSpec(
        source_type="huggingface",
        name="rf-detr-small",
        size="small",
        repo_id="Roboflow/rf-detr-small",
        model_url=f"{_BASE_URL}/Roboflow/rf-detr-small",
        family="RF-DETR",
        training_track="detection",
        class_space="coco",
        training_dataset="COCO",
        checkpoint_format="safetensors",
        inference_engine="transformers",
        note="Detection checkpoint from the official Roboflow Hugging Face organization page.",
    ),
    ModelSpec(
        source_type="huggingface",
        name="rf-detr-base",
        size="base",
        repo_id="Roboflow/rf-detr-base",
        model_url=f"{_BASE_URL}/Roboflow/rf-detr-base",
        family="RF-DETR",
        training_track="detection",
        class_space="coco",
        training_dataset="COCO",
        checkpoint_format="safetensors",
        inference_engine="transformers",
        note="Detection checkpoint from the official Roboflow Hugging Face organization page.",
    ),
    ModelSpec(
        source_type="huggingface",
        name="rf-detr-base-2",
        size="base-2",
        repo_id="Roboflow/rf-detr-base-2",
        model_url=f"{_BASE_URL}/Roboflow/rf-detr-base-2",
        family="RF-DETR",
        training_track="detection",
        class_space="coco",
        training_dataset="COCO",
        checkpoint_format="safetensors",
        inference_engine="transformers",
        note="Detection checkpoint from the official Roboflow Hugging Face organization page.",
    ),
    ModelSpec(
        source_type="huggingface",
        name="rf-detr-medium",
        size="medium",
        repo_id="Roboflow/rf-detr-medium",
        model_url=f"{_BASE_URL}/Roboflow/rf-detr-medium",
        family="RF-DETR",
        training_track="detection",
        class_space="coco",
        training_dataset="COCO",
        checkpoint_format="safetensors",
        inference_engine="transformers",
        note="Detection checkpoint from the official Roboflow Hugging Face organization page.",
    ),
    ModelSpec(
        source_type="huggingface",
        name="rf-detr-large",
        size="large",
        repo_id="Roboflow/rf-detr-large",
        model_url=f"{_BASE_URL}/Roboflow/rf-detr-large",
        family="RF-DETR",
        training_track="detection",
        class_space="coco",
        training_dataset="COCO",
        checkpoint_format="safetensors",
        inference_engine="transformers",
        note="Detection checkpoint from the official Roboflow Hugging Face organization page.",
    ),
)

RFDETR_MODEL_NAMES: tuple[str, ...] = tuple(model.name for model in RFDETR_MODELS)
RFDETR_REPO_IDS: tuple[str, ...] = tuple(model.repo_id for model in RFDETR_MODELS)
RFDETR_MODEL_URLS: tuple[str, ...] = tuple(model.model_url for model in RFDETR_MODELS)
