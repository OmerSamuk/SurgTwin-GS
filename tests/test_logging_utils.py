from pathlib import Path
import json

from surgtwin.training.logging_utils import collect_environment, write_json


def test_collect_environment_returns_dict():
    env = collect_environment()
    assert isinstance(env, dict)
    assert "python_version" in env
    assert "torch_version" in env
    assert "platform" in env
    assert "git_commit" in env
    assert env["backend"] == "gsplat"


def test_collect_environment_cloud_fields_defaults():
    env = collect_environment()
    for field in ["cloud_provider", "zone", "machine_type", "conda_env"]:
        assert field in env, f"Missing {field}"
        assert isinstance(env[field], str)


def test_write_json_creates_file(tmp_path: Path):
    d = {"key": "value", "num": 42}
    p = tmp_path / "test.json"
    write_json(p, d)
    assert p.exists()
    with open(p) as f:
        loaded = json.load(f)
    assert loaded["key"] == "value"
    assert loaded["num"] == 42
