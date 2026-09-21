"""Versioned artifact dependencies and recoverable multi-file publication."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from carvision.utils.paths import repo_root, runs_dir

T = TypeVar("T")

VERSION = 1


def fingerprint(path: Path) -> str:
    """Hash a file without loading a large checkpoint into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    value: dict[str, Any] = json.loads(path.read_text())
    return value


def dependencies(paths: list[Path], command: str) -> dict[str, Any]:
    """Capture required files, including explicitly absent optional inputs."""
    return {
        "version": VERSION,
        "command": command,
        "files": {str(p.resolve()): fingerprint(p) if p.is_file() else None for p in paths},
    }


def validate(record: dict[str, Any]) -> None:
    """Reject missing, changed, or unversioned dependencies."""
    command = record.get("command", "the producing command")
    if record.get("version") != VERSION or not record.get("files"):
        raise ValueError(f"Unverified artifact; rerun {command}.")
    for name, expected in record["files"].items():
        path = Path(name)
        actual = fingerprint(path) if path.is_file() else None
        if actual != expected:
            raise ValueError(f"Stale artifact input {name}; rerun {command}.")


def evaluation_inputs(run: Path, backbone: str, splits: list[str]) -> list[Path]:
    """Collect model, dataset, split and current cache identities."""
    from carvision.data.splits import load_split
    from carvision.features.cache import compute_cache_key, entry_dir
    from carvision.models.backbones import get_backbone

    root = repo_root()
    paths = [
        run / "config.json",
        run / "metrics.json",
        run / "checkpoint.pt",
        run / "test_predictions.npz",
        root / "data/stanford_cars/download.json",
        root / "data/stanford_cars/classes.txt",
        root / "data/splits/provenance.json",
    ]
    for split in splits:
        paths.append(root / f"data/splits/{split}.csv")
        frame = load_split(split)
        key = compute_cache_key(
            get_backbone(backbone), frame.image_id.tolist(), frame.label_id.to_numpy()
        )
        directory = entry_dir(backbone, split, key)
        paths.extend(
            directory / name
            for name in ("manifest.json", "embeddings.npy", "labels.npy", "image_ids.txt")
        )
    return paths


def seal_evaluation(data: dict[str, Any], backbone: str | None = None) -> None:
    """Bind evaluation settings and metrics to their recorded dependency set."""
    if backbone is not None:
        from carvision.models.backbones import get_backbone

        data["provenance"]["backbone"] = backbone
        data["provenance"]["backbone_signature"] = get_backbone(backbone).cache_key()
    payload = {key: value for key, value in data.items() if key != "provenance"}
    data["provenance"]["payload_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def validate_evaluation(run: Path) -> dict[str, Any]:
    """Verify an evaluation against its current dependencies."""
    command = "carvision zeroshot" if run.name.startswith("zeroshot-") else "carvision eval"
    try:
        data = read(run / "evaluation.json")
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Missing or unreadable evaluation for {run.name}; rerun {command}."
        ) from exc
    command = "carvision zeroshot" if data.get("baseline") else "carvision eval"
    record = data.get("provenance", {})
    record.setdefault("command", command)
    validate(record)
    payload = {key: value for key, value in data.items() if key != "provenance"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if record.get("payload_sha256") != digest:
        raise ValueError(f"Evaluation metadata changed; rerun {command}.")
    if record.get("backbone"):
        from carvision.models.backbones import get_backbone

        if get_backbone(record["backbone"]).cache_key() != record.get("backbone_signature"):
            raise ValueError(f"Backbone identity changed; rebuild cache and rerun {command}.")
    if data.get("baseline"):
        from carvision.models.zeroshot import PROMPT_TEMPLATES

        if data.get("prompt_templates") != list(PROMPT_TEMPLATES):
            raise ValueError("Zero-shot prompts changed; rerun carvision zeroshot.")
    if record.get("predictions_sha256") != fingerprint(run / "test_predictions.npz"):
        raise ValueError(f"Stale predictions; rerun {command}.")
    return data


def active_names() -> set[str] | None:
    """Return the active sweep's run names, when defined."""
    path = runs_dir().parent / "sweep_manifest.json"
    return set(read(path)["expected_runs"]) if path.exists() else None


def representative(items: Sequence[T], key: Callable[[T], tuple[float, int]]) -> T:
    """Select the median validation seed with a shared deterministic ordering."""
    return sorted(items, key=key)[len(items) // 2]


def representatives() -> list[Path]:
    """Select median-validation seeds from the active sweep."""
    grouped: dict[tuple[str, str], list[Path]] = {}
    expected = active_names()
    for path in sorted(runs_dir().glob("*/evaluation.json")):
        run = path.parent
        if read(path).get("baseline") or (expected is not None and run.name not in expected):
            continue
        config = read(run / "config.json")
        grouped.setdefault((config["backbone"], config["head"]), []).append(run)
    return [
        representative(
            group,
            key=lambda p: (
                read(p / "metrics.json")["best_val_top1"],
                read(p / "config.json")["seed"],
            ),
        )
        for _, group in sorted(grouped.items())
    ]


def publish(files: dict[Path, bytes], journal: Path) -> None:
    """Publish files with rollback and a persistent interrupted-update journal."""
    journal.parent.mkdir(parents=True, exist_ok=True)
    if journal.exists():
        recover(journal)
    staging = Path(tempfile.mkdtemp(prefix=".publication-", dir=journal.parent))
    entries = []
    try:
        for index, (target, content) in enumerate(files.items()):
            target.parent.mkdir(parents=True, exist_ok=True)
            new, backup = staging / f"{index}.new", staging / f"{index}.old"
            new.write_bytes(content)
            existed = target.exists()
            if existed:
                shutil.copy2(target, backup)
            entries.append(
                {
                    "target": str(target.resolve()),
                    "new": str(new.resolve()),
                    "backup": str(backup.resolve()),
                    "existed": existed,
                }
            )
        journal.write_text(json.dumps({"staging": str(staging.resolve()), "entries": entries}))
        for entry in entries:
            Path(str(entry["new"])).replace(Path(str(entry["target"])))
        journal.unlink()
    except Exception:
        if journal.exists():
            recover(journal)
        raise
    finally:
        if not journal.exists():
            shutil.rmtree(staging, ignore_errors=True)


def recover(journal: Path) -> None:
    """Restore the previous publication after an interrupted replacement."""
    payload = read(journal)
    for entry in payload["entries"]:
        target = Path(entry["target"])
        if entry["existed"]:
            shutil.copy2(entry["backup"], target)
        else:
            target.unlink(missing_ok=True)
    journal.unlink()
    shutil.rmtree(payload["staging"], ignore_errors=True)
