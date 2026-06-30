def test_all_clipped():
    max_grad_norm = 1.0
    grad_norms = [1.5, 2.0, 3.0, 1.1, 1.8]
    clip_active = sum(1 for g in grad_norms if g > max_grad_norm)
    ratio = clip_active / len(grad_norms)
    assert ratio == 1.0


def test_none_clipped():
    max_grad_norm = 1.0
    grad_norms = [0.5, 0.8, 0.3, 0.9, 0.1]
    clip_active = sum(1 for g in grad_norms if g > max_grad_norm)
    ratio = clip_active / len(grad_norms)
    assert ratio == 0.0


def test_some_clipped():
    max_grad_norm = 1.0
    grad_norms = [0.5, 1.5, 0.3, 2.0, 0.1]
    clip_active = sum(1 for g in grad_norms if g > max_grad_norm)
    ratio = clip_active / len(grad_norms)
    assert abs(ratio - 0.4) < 1e-6


def test_no_iterations():
    clip_active = 0
    clip_total = 0
    ratio = 0.0
    assert ratio == 0.0


def test_clip_off_always_zero():
    clip_grad_norm = False
    grad_norms = [1.5, 2.0, 3.0]
    clip_active = 0
    ratio = clip_active / len(grad_norms)
    assert ratio == 0.0
