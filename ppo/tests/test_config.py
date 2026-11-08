from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ppo.config import Config, dump, load, resolve, score
from ppo.train import _is_better_eval

REPO = Path(__file__).resolve().parents[2]
EMPTY = REPO / "ppo" / "configs" / "empty.yaml"


def test_resolve_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ppo.config as config

    monkeypatch.setattr(config, "REPO", tmp_path)
    run_cfg = tmp_path / "outputs" / "trained_agents" / "some-run" / "config.yaml"
    run_cfg.parent.mkdir(parents=True)
    run_cfg.write_text("run: 1\n")
    assert resolve("some-run") == run_cfg


def test_override_unknown_key_raises() -> None:
    with pytest.raises(KeyError):
        load(EMPTY, {"ppo.not_a_real_key": 1.0})


def test_map_override_switches_track(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ppo.config as config

    monkeypatch.setattr(config, "REPO", tmp_path)
    track_dir = tmp_path / "sf" / "tracks" / "other"
    track_dir.mkdir(parents=True)
    (track_dir / "track.yaml").write_text(
        "map: other\nopt_time: 9.5\nlength: 30.0\ndirection: ccw\nstart_pose: [1.0, 2.0, 0.5]\n"
    )
    cfg = load(EMPTY, {"track.map": "other", "ppo.alpha": 0.001})
    assert cfg.track.map == "other"
    assert cfg.track.opt_time == pytest.approx(9.5)
    assert cfg.track.length == pytest.approx(30.0)
    assert cfg.ppo.alpha == pytest.approx(0.001)


def test_best_eval_follows_score_not_lexicographic() -> None:
    """A slight success drop with a larger pace gain wins under score()."""
    lex_winner = (0.80, 0.90)
    score_winner = (0.79, 1.05)
    assert lex_winner > score_winner
    assert score(*score_winner) > score(*lex_winner)
    assert _is_better_eval(score_winner, lex_winner)
    assert not _is_better_eval(lex_winner, score_winner)
    assert _is_better_eval(lex_winner, None)


@pytest.mark.parametrize(
    ("layout_switch", "redrawn"),
    [("complete", {1}), ("episode", {0, 1}), ("never", set())],
)
def test_layout_switch_rule(
    layout_switch: str, redrawn: set[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env 0 crashed, env 1 completed, env 2 still running."""
    import numpy as np

    import ppo.train as train

    cfg = load(EMPTY, {"training.layout_switch": layout_switch, "training.n_empty_envs": 0})
    assigned: list[int] = []
    monkeypatch.setattr(
        train, "_assign_layout", lambda venv, cfg, track, zones, i, rng: assigned.append(i)
    )

    class Venv:
        obs = np.zeros((train.N_PARALLEL_ENV, 1))

    n = train.N_PARALLEL_ENV
    dones = [True, True, False] + [False] * (n - 3)
    infos = [{"complete": False}, {"complete": True}, {}] + [{}] * (n - 3)
    train._maybe_switch_finished_layouts(
        train.PhaseState(n_obstacles=2), Venv(), cfg, None, None, None,
        dones, infos, np.zeros((n, 1)),
    )
    assert set(assigned) == redrawn


def test_layout_switch_rejects_unknown() -> None:
    with pytest.raises(Exception):
        load(EMPTY, {"training.layout_switch": "sometimes"})


def test_dump_round_trip() -> None:
    cfg = load(EMPTY, {"training.n_phase1_steps": 12345})
    again = Config(**yaml.safe_load(dump(cfg)))
    assert again.model_dump() == cfg.model_dump()
    assert again.training.n_phase1_steps == 12345
