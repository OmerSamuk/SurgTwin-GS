from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from surgtwin.cameras.camera_types import CameraData


@dataclass(frozen=True)
class FrameSample:
    sample_id: str
    sequence_id: str
    frame_index: int
    left_rgb_path: Path
    right_rgb_path: Path
    left_depth_path: Path
    right_depth_path: Optional[Path]
    left_camera: CameraData
    right_camera: CameraData
    left_tool_mask_path: Optional[Path]
    right_tool_mask_path: Optional[Path]
    left_specular_mask_path: Optional[Path]
    right_specular_mask_path: Optional[Path]
    split: str
