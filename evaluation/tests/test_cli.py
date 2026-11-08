"""CLI checkpoint suffix, seed discovery, and map check."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.cli import _checkpoint_suffix, _discover_seeds, main


def test_checkpoint_suffix_mapping() -> None:
    assert _checkpoint_suffix("best_eval") == "_best_eval"
    assert _checkpoint_suffix("final") == ""
    assert _checkpoint_suffix("1000000") == "_ckpt_1000000"
    assert _checkpoint_suffix("5000000") == "_ckpt_5000000"
    with pytest.raises(ValueError):
        _checkpoint_suffix("best")


def test_discover_seeds_from_run_dir(tmp_path: Path) -> None:
    (tmp_path / "_actor_seed_17_best_eval").write_text("x")
    (tmp_path / "_actor_seed_76_best_eval").write_text("x")
    (tmp_path / "_actor_seed_17").write_text("x")
    (tmp_path / "_actor_seed_122_ckpt_1000").write_text("x")
    (tmp_path / "_actor_seed_693_best_eval").write_text("x")
    assert _discover_seeds(tmp_path, _checkpoint_suffix("best_eval")) == [17, 76, 693]
    assert _discover_seeds(tmp_path, _checkpoint_suffix("final")) == [17]
    assert _discover_seeds(tmp_path, _checkpoint_suffix("1000")) == [122]


def test_eval_set_map_mismatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("eval should exit before running layouts")

    other = tmp_path / "other-map-set"
    other.mkdir()
    (other / "k0.json").write_text(
        '{"map": "other-map", "set": "other-map-set", "index": 0, "obstacles": []}'
    )
    monkeypatch.setattr("ppo.layouts._LAYOUTS", tmp_path)
    monkeypatch.setattr("evaluation.cli._run_policy", _boom)
    monkeypatch.setattr("evaluation.cli._run_filter", _boom)
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "empty",
                "--eval-set",
                "other-map-set",
                "--seeds",
                "17",
                "--layout",
                "0",
                "--jobs",
                "1",
            ]
        )
    captured = capsys.readouterr()
    text = f"{caught.value}\n{captured.out}\n{captured.err}".lower()
    assert caught.value.code not in (0, None)
    assert "map" in text
