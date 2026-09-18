"""Configuration must actually be read.

This project previously shipped a `configs/` tree that nothing loaded, while the
download module's docstring told users to edit it to change the dataset mirror. Editing
it did nothing. These tests hold the fix in place.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from carvision.config import (
    ConfigError,
    DataConfig,
    available_data_configs,
    config_dir,
    load_data_config,
)


def test_the_shipped_config_loads() -> None:
    config = load_data_config("stanford_cars")
    assert config.num_classes == 196
    assert config.hf_repo_id


def test_editing_the_mirror_actually_changes_what_download_would_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact scenario that used to silently do nothing."""
    monkeypatch.setenv("CARVISION_ROOT", str(tmp_path))
    directory = tmp_path / "configs" / "data"
    directory.mkdir(parents=True)
    (directory / "stanford_cars.yaml").write_text(
        "num_classes: 196\nhf_repo_id: someone/my-mirror\nimage_column: img\n"
    )

    config = load_data_config()
    assert config.hf_repo_id == "someone/my-mirror"

    from carvision.data.download import resolve_mirror

    spec = resolve_mirror(config.hf_repo_id, **config.mirror_overrides())
    assert spec.repo_id == "someone/my-mirror"
    assert spec.image_column == "img"


def test_a_typo_is_an_error_not_a_silent_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A misspelled key must fail loudly; quietly ignoring it is the original bug."""
    monkeypatch.setenv("CARVISION_ROOT", str(tmp_path))
    directory = tmp_path / "configs" / "data"
    directory.mkdir(parents=True)
    (directory / "typo.yaml").write_text("hf_repoid: someone/mirror\n")

    with pytest.raises(ConfigError, match="unknown key"):
        load_data_config("typo")


def test_a_missing_config_lists_what_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CARVISION_ROOT", str(tmp_path))
    (tmp_path / "configs" / "data").mkdir(parents=True)
    (tmp_path / "configs" / "data" / "real.yaml").write_text("num_classes: 3\n")

    with pytest.raises(ConfigError, match="real"):
        load_data_config("nope")


def test_defaults_round_trip() -> None:
    config = DataConfig()
    assert set(config.mirror_overrides()) == {
        "image_column",
        "label_column",
        "train_split",
        "test_split",
    }


def test_shipped_configs_are_discoverable() -> None:
    assert "stanford_cars" in available_data_configs()
    assert config_dir().is_dir()


def test_no_stale_backbone_or_head_configs_remain() -> None:
    """Those directories duplicated the registry and were removed; keep them gone."""
    assert not (config_dir() / "backbone").exists()
    assert not (config_dir() / "head").exists()


def test_nothing_imports_hydra() -> None:
    """hydra-core was a declared dependency that no module used. Keep it that way."""
    import subprocess

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["grep", "-rn", "--include=*.py", "-E", r"^\s*(import|from)\s+hydra", "src", "app"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout == "", f"hydra is imported after all:\n{result.stdout}"
