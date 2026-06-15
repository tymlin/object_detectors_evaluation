from pathlib import Path

import pytest
from pydantic import ValidationError

from object_detectors_evaluation.datasets import DetectionDatasetConfig
from object_detectors_evaluation.utils.files import load_yaml


def test_detection_dataset_config_accepts_required_fields(tmp_path: Path) -> None:
    dataset_dirpath = tmp_path / "coco-2017"

    config = DetectionDatasetConfig(
        dataset_dirpath=dataset_dirpath,
        split="validation",
    )

    assert config.dataset_dirpath == dataset_dirpath
    assert config.split == "validation"
    assert config.classes_of_interest is None
    assert config.include_crowd is True
    assert config.drop_images_with_crowd is False
    assert config.remove_empty_images is False


def test_detection_dataset_config_accepts_filtering_fields(tmp_path: Path) -> None:
    config = DetectionDatasetConfig(
        dataset_dirpath=tmp_path / "open-images-v7",
        split="test",
        classes_of_interest=["Person", "Car"],
        include_crowd=False,
        drop_images_with_crowd=True,
        remove_empty_images=True,
    )

    assert config.classes_of_interest == ["Person", "Car"]
    assert config.include_crowd is False
    assert config.drop_images_with_crowd is True
    assert config.remove_empty_images is True


def test_detection_dataset_config_rejects_unknown_split(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DetectionDatasetConfig(
            dataset_dirpath=tmp_path / "dataset",
            split="dev",
        )


def test_detection_dataset_config_rejects_extra_fields(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        DetectionDatasetConfig.model_validate(
            {
                "dataset_dirpath": tmp_path / "dataset",
                "split": "validation",
                "unknown": "value",
            }
        )


def test_detection_dataset_config_loads_from_yaml_file(tmp_path: Path) -> None:
    config_filepath = tmp_path / "dataset.yaml"
    dataset_dirpath = tmp_path / "coco-2017"
    config_filepath.write_text(
        "\n".join(
            [
                f"dataset_dirpath: {dataset_dirpath}",
                "split: validation",
                "classes_of_interest:",
                "  - person",
                "include_crowd: true",
                "drop_images_with_crowd: false",
                "remove_empty_images: true",
            ]
        )
    )

    config_data = load_yaml(config_filepath)
    config = DetectionDatasetConfig.model_validate(config_data)

    assert config.dataset_dirpath == dataset_dirpath
    assert config.split == "validation"
    assert config.classes_of_interest == ["person"]
    assert config.include_crowd is True
    assert config.drop_images_with_crowd is False
    assert config.remove_empty_images is True
