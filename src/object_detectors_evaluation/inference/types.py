from typing import Literal, TypeAlias

import numpy as np
import torch

InferenceDevice = Literal["auto", "cpu", "cuda", "mps"]
InferenceDType = Literal["auto", "float32", "float16", "bfloat16"]
ImageId: TypeAlias = int | str | None
ImageInput: TypeAlias = np.ndarray | torch.Tensor
ProcessedInputs: TypeAlias = list[ImageInput] | torch.Tensor | dict[str, torch.Tensor]
