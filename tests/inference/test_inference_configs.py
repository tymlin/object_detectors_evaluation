from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from object_detectors_evaluation.inference import DetectionInferenceConfig
from object_detectors_evaluation.utils.files import load_yaml


def test_detection_inference_config_defaults() -> None:
    config = DetectionInferenceConfig()

    assert config.device == "auto"
    assert config.score_threshold == 0.0
    assert config.max_detections is None
    assert config.dtype is None


def test_detection_inference_config_accepts_runtime_fields() -> None:
    config = DetectionInferenceConfig(
        device="cpu",
        score_threshold=0.25,
        max_detections=100,
        dtype="float32",
    )

    assert config.device == "cpu"
    assert config.score_threshold == 0.25
    assert config.max_detections == 100
    assert config.dtype == "float32"


@pytest.mark.parametrize(
    "config_data",
    [
        {"score_threshold": -0.1},
        {"score_threshold": 1.1},
        {"max_detections": 0},
    ],
)
def test_detection_inference_config_rejects_invalid_numeric_values(config_data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        DetectionInferenceConfig.model_validate(config_data)


def test_detection_inference_config_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        DetectionInferenceConfig.model_validate(
            {
                "device": "cpu",
                "unknown": "value",
            }
        )


def test_detection_inference_config_loads_from_yaml_file(tmp_path: Path) -> None:
    config_filepath = tmp_path / "inference.yaml"
    config_filepath.write_text(
        "\n".join(
            [
                "device: cpu",
                "score_threshold: 0.5",
                "max_detections: 25",
                "dtype: float32",
            ]
        )
    )

    config_data = load_yaml(config_filepath)
    config = DetectionInferenceConfig.model_validate(config_data)

    assert config.device == "cpu"
    assert config.score_threshold == 0.5
    assert config.max_detections == 25
    assert config.dtype == "float32"
