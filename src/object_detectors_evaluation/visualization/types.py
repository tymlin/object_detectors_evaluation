from collections.abc import Mapping
from typing import TypeAlias

Color: TypeAlias = str | tuple[float, float, float] | tuple[float, float, float, float]
ColorKey: TypeAlias = int | str
ColorMap: TypeAlias = Mapping[ColorKey, Color]
