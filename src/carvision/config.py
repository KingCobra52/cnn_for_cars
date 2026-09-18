"""Configuration that is genuinely configuration.

The split is deliberate, and it is the fix for a real defect: this project previously
shipped a `configs/` tree that **nothing read**. The download module's docstring told
you to edit `configs/data/stanford_cars.yaml` to change the dataset mirror, and editing
it did nothing at all.

The rule now:

* **Dataset acquisition is configuration.** Which Hub mirror, which columns, which split
  names, the validation fraction and its seed are all things a user legitimately changes
  without touching code. They live in ``configs/data/*.yaml`` and this module loads them.
* **Backbones and heads are code.** A backbone's embedding width and preprocessing are
  properties of the pretrained weights, not preferences, and the registry has to hold a
  factory function regardless. Duplicating the width into YAML would create a second
  source of truth that can only ever drift out of step with the first. So
  ``carvision.models.backbones.REGISTRY`` is the single source, and the former
  ``configs/backbone/`` and ``configs/head/`` files are gone.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from carvision.utils.logging import get_logger
from carvision.utils.paths import repo_root

logger = get_logger(__name__)

DEFAULT_DATA_CONFIG = "stanford_cars"


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing or malformed."""


@dataclass(frozen=True)
class DataConfig:
    """How to acquire and split a dataset.

    Attributes:
        name: Config name, matching the file stem.
        num_classes: Expected number of classes, checked at download time.
        hf_repo_id: Hugging Face Hub dataset repository.
        hf_revision: Optional Hub revision to pin.
        image_column: Column holding the image.
        label_column: Column holding the integer class id.
        train_split: Name of the mirror's training split.
        test_split: Name of the mirror's test split.
        val_fraction: Fraction of the official train split held out for validation.
        split_seed: Seed for the stratified validation draw.
    """

    name: str = DEFAULT_DATA_CONFIG
    num_classes: int = 196
    hf_repo_id: str = "tanganke/stanford_cars"
    hf_revision: str | None = None
    image_column: str = "image"
    label_column: str = "label"
    train_split: str = "train"
    test_split: str = "test"
    val_fraction: float = 0.15
    split_seed: int = 20260918

    def mirror_overrides(self) -> dict[str, str]:
        """Return the fields that override a :class:`~carvision.data.download.MirrorSpec`."""
        return {
            "image_column": self.image_column,
            "label_column": self.label_column,
            "train_split": self.train_split,
            "test_split": self.test_split,
        }


def config_dir() -> Path:
    """Directory holding the YAML configuration files."""
    return repo_root() / "configs"


def available_data_configs() -> list[str]:
    """Return the names of the data configs that exist on disk."""
    directory = config_dir() / "data"
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.yaml"))


def load_data_config(name: str = DEFAULT_DATA_CONFIG) -> DataConfig:
    """Read ``configs/data/<name>.yaml``.

    Args:
        name: Config name, matching the file stem.

    Returns:
        The parsed configuration.

    Raises:
        ConfigError: If the file is absent, is not a mapping, or contains a key that is
            not a field of :class:`DataConfig`. An unknown key is an error rather than a
            warning precisely because the old silent-no-op behaviour is what this module
            exists to prevent -- a typo must not look like it worked.
    """
    from omegaconf import OmegaConf

    path = config_dir() / "data" / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(
            f"No data config at {path}. Available: {available_data_configs() or 'none'}"
        )

    loaded: Any = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a mapping, got {type(loaded).__name__}.")

    known = {field.name for field in fields(DataConfig)}
    unknown = set(loaded) - known
    if unknown:
        raise ConfigError(
            f"{path} has unknown key(s) {sorted(unknown)}. Known keys: {sorted(known)}."
        )

    logger.debug("Loaded data config %s from %s", name, path)
    return DataConfig(**{**loaded, "name": name})
