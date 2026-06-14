from typing import Any

import numpy as np
import pytest
import torch

from object_detectors_evaluation.inference.base import BaseDetectionInferenceEngine, DetectionInputBatch
from object_detectors_evaluation.inference.engines.ultralytics import UltralyticsDetectionInferenceEngine
from object_detectors_evaluation.inference.predictions import DetectionPredictionBatch


class FakeDetectionInferenceEngine(BaseDetectionInferenceEngine):
    """Minimal concrete engine for testing base preprocessing only."""

    def load_model(self) -> Any:
        return None

    def predict(self, preprocessed_batch: DetectionInputBatch) -> Any:
        return None

    def postprocess(self, preprocessed_batch: DetectionInputBatch, raw_outputs: Any) -> DetectionPredictionBatch:
        return DetectionPredictionBatch(predictions=())


def test_base_preprocess_accepts_numpy_hwc_images() -> None:
    engine = object.__new__(FakeDetectionInferenceEngine)
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    input_batch = engine.preprocess(images=[image], image_ids=["image-1"])

    assert len(input_batch.processed_inputs) == 1
    assert input_batch.processed_inputs[0] is image
    assert input_batch.image_ids == ["image-1"]
    assert input_batch.image_sizes == [(10, 20)]


def test_base_preprocess_accepts_torch_chw_images() -> None:
    engine = object.__new__(FakeDetectionInferenceEngine)
    image = torch.zeros((3, 10, 20), dtype=torch.float32)

    input_batch = engine.preprocess(images=[image], image_ids=["image-1"])

    assert len(input_batch.processed_inputs) == 1
    assert input_batch.processed_inputs[0] is image
    assert input_batch.image_ids == ["image-1"]
    assert input_batch.image_sizes == [(10, 20)]


@pytest.mark.parametrize(
    "images",
    [
        np.zeros((10, 20, 3), dtype=np.uint8),
        torch.zeros((3, 10, 20), dtype=torch.float32),
    ],
)
def test_base_preprocess_rejects_single_image_without_batch_container(images: np.ndarray | torch.Tensor) -> None:
    engine = object.__new__(FakeDetectionInferenceEngine)

    with pytest.raises(TypeError, match="sequence of per-image inputs"):
        engine.preprocess(images=images)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((10, 20), dtype=np.uint8),
        np.zeros((1, 10, 20, 3), dtype=np.uint8),
        torch.zeros((10, 20), dtype=torch.float32),
        torch.zeros((1, 3, 10, 20), dtype=torch.float32),
    ],
)
def test_base_preprocess_rejects_invalid_image_shapes(image: np.ndarray | torch.Tensor) -> None:
    engine = object.__new__(FakeDetectionInferenceEngine)

    with pytest.raises(ValueError):
        engine.preprocess(images=[image])


def test_base_preprocess_rejects_mismatched_image_ids() -> None:
    engine = object.__new__(FakeDetectionInferenceEngine)
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Expected `1` image ids, got `2`"):
        engine.preprocess(images=[image], image_ids=["image-1", "image-2"])


def test_ultralytics_preprocess_converts_numpy_rgb_to_bgr() -> None:
    engine = object.__new__(UltralyticsDetectionInferenceEngine)
    image = np.asarray([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)

    input_batch = engine.preprocess(images=[image], image_ids=["image-1"])

    assert isinstance(input_batch.processed_inputs, list)
    assert input_batch.processed_inputs[0].tolist() == [[[3, 2, 1], [6, 5, 4]]]
    assert input_batch.image_ids == ["image-1"]
    assert input_batch.image_sizes == [(1, 2)]


def test_ultralytics_preprocess_stacks_torch_chw_images() -> None:
    engine = object.__new__(UltralyticsDetectionInferenceEngine)
    image_1 = torch.zeros((3, 10, 20), dtype=torch.float32)
    image_2 = torch.ones((3, 10, 20), dtype=torch.float32)

    input_batch = engine.preprocess(images=[image_1, image_2], image_ids=["image-1", "image-2"])

    assert isinstance(input_batch.processed_inputs, torch.Tensor)
    assert input_batch.processed_inputs.shape == (2, 3, 10, 20)
    assert torch.equal(input_batch.processed_inputs[0], image_1)
    assert torch.equal(input_batch.processed_inputs[1], image_2)
    assert input_batch.image_ids == ["image-1", "image-2"]
    assert input_batch.image_sizes == [(10, 20), (10, 20)]
