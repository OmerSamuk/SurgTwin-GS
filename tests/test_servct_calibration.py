import numpy as np
import torch
from surgtwin.data.servct_calibration import parse_rectified_calibration, rectified_to_camera_data


def test_parse_rectified_calibration(tmp_path):
    calib = {
        "P1": [[500.0, 0.0, 360.0, 0.0], [0.0, 500.0, 288.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        "P2": [[500.0, 0.0, 360.0, -2500.0], [0.0, 500.0, 288.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        "Q": [[1.0, 0.0, 0.0, -360.0], [0.0, 1.0, 0.0, -288.0], [0.0, 0.0, 0.0, 500.0], [0.0, 0.0, 5.0, 0.0]],
    }
    import json
    p = tmp_path / "Calibration.json"
    with open(p, "w") as f:
        json.dump(calib, f)

    P1, P2, Q = parse_rectified_calibration(p)
    assert P1.shape == (3, 4)
    assert P2.shape == (3, 4)
    assert Q.shape == (4, 4)


def test_rectified_to_camera_data():
    P1 = np.array([[500.0, 0.0, 360.0, 0.0], [0.0, 500.0, 288.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    baseline_mm = 5.0
    P2 = np.array([[500.0, 0.0, 360.0, -500.0 * baseline_mm], [0.0, 500.0, 288.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    Q = np.eye(4, dtype=np.float64)

    result = rectified_to_camera_data(P1, P2, Q, 576, 720)

    assert result["K_left"].shape == (3, 3)
    assert result["K_right"].shape == (3, 3)
    assert result["c2w_left"].shape == (4, 4)
    assert result["c2w_right"].shape == (4, 4)
    assert result["w2c_left"].shape == (4, 4)
    assert result["w2c_right"].shape == (4, 4)

    assert torch.allclose(result["c2w_left"], torch.eye(4), atol=1e-6)
    assert abs(result["c2w_right"][0, 3].item() - baseline_mm * 0.001) < 1e-6

    E = result["w2c_left"] @ result["c2w_left"] - torch.eye(4)
    assert E[:3, :3].abs().max().item() < 1e-6
    assert E[:3, 3].abs().max().item() < 1e-6


def test_rectified_inverse_consistency():
    P1 = np.array([[500.0, 0.0, 360.0, 0.0], [0.0, 500.0, 288.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    P2 = P1.copy()
    P2[0, 3] = -2500.0
    Q = np.eye(4, dtype=np.float64)
    result = rectified_to_camera_data(P1, P2, Q, 576, 720)

    for side in ["left", "right"]:
        E = result[f"w2c_{side}"] @ result[f"c2w_{side}"] - torch.eye(4)
        rot_err = E[:3, :3].abs().max().item()
        trans_err = E[:3, 3].abs().max().item()
        assert rot_err < 1e-4, f"{side} rotation error: {rot_err}"
        assert trans_err < 1e-3, f"{side} translation error: {trans_err}"


def test_baseline_extraction():
    P1 = np.array([[500.0, 0.0, 360.0, 0.0], [0.0, 500.0, 288.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    P2 = np.array([[500.0, 0.0, 360.0, -2500.0], [0.0, 500.0, 288.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    Q = np.eye(4, dtype=np.float64)
    result = rectified_to_camera_data(P1, P2, Q, 576, 720)

    baseline_m = result["c2w_right"][0, 3].item()
    assert abs(baseline_m - 5.0 * 0.001) < 1e-6, f"Baseline mismatch: {baseline_m}"
