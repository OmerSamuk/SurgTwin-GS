import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def parse_rectified_calibration(json_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(json_path, "r") as f:
        calib = json.load(f)
    P1 = np.array(calib["P1"], dtype=np.float64)
    P2 = np.array(calib["P2"], dtype=np.float64)
    Q = np.array(calib["Q"], dtype=np.float64)
    if P1.shape != (3, 4) or P2.shape != (3, 4):
        raise ValueError(f"P1 and P2 must be 3x4 matrices, got P1 {P1.shape}, P2 {P2.shape}")
    if Q.shape != (4, 4):
        raise ValueError(f"Q must be a 4x4 matrix, got {Q.shape}")
    return P1, P2, Q


def rectified_to_camera_data(
    P1: np.ndarray, P2: np.ndarray, Q: np.ndarray, height: int, width: int
) -> dict:
    K_left = P1[:3, :3].copy()
    K_left = K_left.astype(np.float32)

    K_right = P2[:3, :3].copy()
    K_right = K_right.astype(np.float32)

    baseline_mm = -P2[0, 3] / P2[0, 0]
    baseline_m = baseline_mm * 0.001

    c2w_left = np.eye(4, dtype=np.float32)
    w2c_left = np.eye(4, dtype=np.float32)

    c2w_right = np.eye(4, dtype=np.float32)
    c2w_right[0, 3] = baseline_m

    w2c_right = np.eye(4, dtype=np.float32)
    w2c_right[0, 3] = -baseline_m

    return {
        "K_left": torch.tensor(K_left),
        "K_right": torch.tensor(K_right),
        "c2w_left": torch.tensor(c2w_left),
        "c2w_right": torch.tensor(c2w_right),
        "w2c_left": torch.tensor(w2c_left),
        "w2c_right": torch.tensor(w2c_right),
        "Q": torch.tensor(Q.astype(np.float32)),
    }
