"""CLI wiring. Parsing only -- no subcommand here touches the network or the dataset."""

from __future__ import annotations

import pytest

from carvision.cli import build_parser


def test_version_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--version"])
    assert excinfo.value.code == 0


def test_a_bare_invocation_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_cache_build_requires_a_backbone() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["cache", "build"])


def test_cache_build_defaults_to_all_three_splits() -> None:
    args = build_parser().parse_args(["cache", "build", "--backbone", "resnet50"])
    assert args.splits == ["train", "val", "test"]


def test_train_defaults() -> None:
    args = build_parser().parse_args(["train"])
    assert (args.backbone, args.head, args.seed) == ("dinov2_vits14", "linear", 0)


def test_train_rejects_an_unknown_head() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--head", "transformer"])


def test_sweep_defaults_to_thirty_runs() -> None:
    args = build_parser().parse_args(["sweep"])
    assert len(args.backbones) * len(args.heads) * len(args.seeds) == 30


def test_split_overwrite_is_off_by_default() -> None:
    """Regenerating committed splits invalidates published numbers, so it is opt-in."""
    assert build_parser().parse_args(["data", "split"]).overwrite is False


def test_download_accepts_mirror_overrides() -> None:
    args = build_parser().parse_args(
        ["data", "download", "--repo-id", "someone/mirror", "--image-column", "img"]
    )
    assert (args.repo_id, args.image_column) == ("someone/mirror", "img")
