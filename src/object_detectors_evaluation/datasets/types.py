from typing import Any, Literal, TypeAlias

DatasetName = Literal["coco", "open_images"]
DatasetSplit = Literal["train", "validation", "test"]
DetectionTarget: TypeAlias = dict[str, Any]
