from types import SimpleNamespace

import numpy as np
import torch

from object_detectors_evaluation.inference.engines import RFDetrDetectionInferenceEngine


def _create_engine(
    max_detections: int | None = None,
    id2label: dict[int, str] | None = None,
) -> RFDetrDetectionInferenceEngine:
    if id2label is None:
        id2label = {
            0: "N/A",
            1: "person",
            2: "bicycle",
            3: "N/A",
            4: "car",
        }

    engine = object.__new__(RFDetrDetectionInferenceEngine)
    engine.model = SimpleNamespace(config=SimpleNamespace(id2label=id2label))
    engine.model_artifact = SimpleNamespace(spec=SimpleNamespace(class_space="coco"))
    engine.max_detections = max_detections
    return engine


def test_rf_detr_class_map_excludes_unused_slots_and_compacts_labels() -> None:
    engine = _create_engine()

    assert engine.class_id_to_name == {0: "person", 1: "bicycle", 2: "car"}


def test_rf_detr_class_map_matches_canonical_coco_labels() -> None:
    unused_class_ids = {0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83}
    id2label = {class_id: "N/A" if class_id in unused_class_ids else f"class-{class_id}" for class_id in range(91)}
    id2label.update({1: "person", 13: "stop sign", 62: "chair", 72: "tv", 86: "vase", 90: "toothbrush"})
    engine = _create_engine(id2label=id2label)

    assert len(engine.class_id_to_name) == 80
    assert engine.class_id_to_name[0] == "person"
    assert engine.class_id_to_name[11] == "stop sign"
    assert engine.class_id_to_name[56] == "chair"
    assert engine.class_id_to_name[62] == "tv"
    assert engine.class_id_to_name[75] == "vase"
    assert engine.class_id_to_name[79] == "toothbrush"


def test_rf_detr_prediction_filters_unused_slots_and_preserves_alignment() -> None:
    engine = _create_engine()
    result = {
        "boxes": torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 2.0, 2.0],
                [2.0, 2.0, 3.0, 3.0],
                [3.0, 3.0, 4.0, 4.0],
            ]
        ),
        "scores": torch.tensor([0.9, 0.8, 0.7, 0.6]),
        "labels": torch.tensor([0, 1, 3, 4]),
    }

    prediction = engine._prediction_from_result(result=result, image_id="image-1", image_size=(10, 20))

    assert prediction.boxes.tolist() == [[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]]
    assert np.allclose(prediction.scores, [0.8, 0.6])
    assert prediction.labels.tolist() == [0, 2]
    assert prediction.labels_names == ("person", "car")


def test_rf_detr_prediction_applies_max_detections_after_filtering() -> None:
    engine = _create_engine(max_detections=1)
    result = {
        "boxes": torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 2.0, 2.0],
                [2.0, 2.0, 3.0, 3.0],
            ]
        ),
        "scores": torch.tensor([0.9, 0.8, 0.7]),
        "labels": torch.tensor([0, 1, 2]),
    }

    prediction = engine._prediction_from_result(result=result, image_id="image-1", image_size=(10, 20))

    assert prediction.boxes.tolist() == [[1.0, 1.0, 2.0, 2.0]]
    assert np.allclose(prediction.scores, [0.8])
    assert prediction.labels.tolist() == [0]
    assert prediction.labels_names == ("person",)
