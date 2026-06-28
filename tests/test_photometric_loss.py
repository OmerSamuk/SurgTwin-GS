import torch

from surgtwin.losses.photometric import photometric_l1


def test_identical_images():
    img = torch.rand(64, 64, 3)
    loss = photometric_l1(img, img)
    assert abs(loss.item()) < 1e-6


def test_known_l1_value():
    pred = torch.zeros(4, 4, 3)
    gt = torch.ones(4, 4, 3)
    loss = photometric_l1(pred, gt)
    assert abs(loss.item() - 1.0) < 1e-6


def test_shape_mismatch():
    try:
        photometric_l1(torch.rand(4, 4, 3), torch.rand(5, 5, 3))
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_alpha_channel_stripped():
    pred = torch.zeros(4, 4, 4)
    gt = torch.ones(4, 4, 4)
    loss = photometric_l1(pred, gt)
    assert abs(loss.item() - 1.0) < 1e-6
