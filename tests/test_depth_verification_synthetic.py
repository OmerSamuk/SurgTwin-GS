import pytest
import torch


def test_synthetic_verify_imports():
    from scripts.verify_depth_synthetic import main as syn_main
    assert callable(syn_main)


def test_synthetic_run_requires_cuda(monkeypatch):
    from scripts.verify_depth_synthetic import main as syn_main
    monkeypatch.setattr("sys.argv", ["verify_depth_synthetic.py", "--output_dir", "tmp"])
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is required"):
            syn_main()


@pytest.mark.skip(reason="Full pipeline test requires CUDA + gsplat rendering; run scripts/verify_depth_synthetic.py directly")
def test_synthetic_full_pipeline():
    pass
