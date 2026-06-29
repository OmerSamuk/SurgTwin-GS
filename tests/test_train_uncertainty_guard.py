from pathlib import Path

from scripts.train_uncertainty import _resolve_output_dir


def test_resolve_creates_new_dir(tmp_path):
    d = tmp_path / "new_dir"
    result = _resolve_output_dir(d, allow_mock=False, overwrite=False)
    assert result == d
    assert d.exists()
    assert list(d.iterdir()) == []


def test_resolve_empty_dir_ok(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    result = _resolve_output_dir(d, allow_mock=False, overwrite=False)
    assert result == d


def test_resolve_non_empty_raises(tmp_path):
    d = tmp_path / "nonempty"
    d.mkdir()
    (d / "dummy.txt").write_text("hello")
    import pytest
    with pytest.raises(SystemExit, match="output_dir is not empty"):
        _resolve_output_dir(d, allow_mock=False, overwrite=False)


def test_resolve_overwrite_cleans(tmp_path):
    d = tmp_path / "to_overwrite"
    d.mkdir()
    (d / "old.txt").write_text("old")
    result = _resolve_output_dir(d, allow_mock=False, overwrite=True)
    assert result == d
    assert list(d.iterdir()) == []


def test_resolve_mock_dir(tmp_path):
    d = tmp_path / "prod"
    debug = _resolve_output_dir(d, allow_mock=True, overwrite=False)
    assert debug.name == f"_debug_prod"
    assert debug.parent == d.parent
    assert debug.exists()
