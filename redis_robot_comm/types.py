"""Type definitions for redis_robot_comm package."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt

# Image types
# Use Any for the inner type to avoid mypy issues with numpy.typing in some environments
ImageArray = npt.NDArray[Any]
ImageMetadata = Dict[str, Any]
ImageResult = Tuple[ImageArray, ImageMetadata]

# Object detection types
ObjectDict = Dict[str, Any]
CameraPose = Dict[str, float]
StreamID = str

# Text overlay types
TextOverlayDict = Dict[str, Any]
LabelList = List[str]
