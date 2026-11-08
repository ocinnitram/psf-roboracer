"""Sweep override flattening and select ranking."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import ppo.select as select_module
from ppo.config import SCORE_PACE_WEIGHT
from ppo.select import (
    FIXED_COLUMNS,
    Candidate,
    candidate_config,
    phase_of,
    rank,
    tsv_columns,
    write_tsv,
)
from ppo.sweep import load_sweep_yaml, overrides_from_wandb


def test_overrides_from_wandb_config_dict() -> None:
    raw: dict[str, Any] = {
        "ppo.alpha": 0.0005,
        "ppo.clip": 0.2,
        "ppo.n_epochs": 8,
        "_wandb": {"foo": 1},
        "wandb_version": 1,
    }
    assert overrides_from_wandb(raw) == {
        "ppo.alpha": 0.0005,
        "ppo.clip": 0.2,
        "ppo.n_epochs": 8,
    }


def test_overrides_from_nested_wandb_config() -> None:
    raw = {"ppo": {"alpha": 1.0e-4, "clip": 0.25}, "_wandb": {"ignored": True}}
    assert overrides_from_wandb(raw) == {"ppo.alpha": 1.0e-4, "ppo.clip": 0.25}


def _evals(*scores: float) -> tuple[dict[str, Any], ...]:
    """Evaluation rows with score, success rate 1.0, and pace derived from score."""
    return tuple(
        {
            "score": s,
            "Eval/SuccessRate": 1.0,
            "Eval/Pace": (s - 1.0) / SCORE_PACE_WEIGHT,
            "Eval/SteerRate": 0.1,
            "Eval/SpeedCmdRate": 0.2,
        }
        for s in scores
    )


def test_rank_uses_mean_of_last_k_not_max_or_last(tmp_path: Any) -> None:
    keys = ["ppo.alpha", "ppo.clip"]
    candidates = [
        # Highest single eval and highest last eval, but the tail averages lowest.
        Candidate(
            "spiky",
            {"ppo.alpha": 1e-4, "ppo.clip": 0.1},
            (_evals(1.02, 1.10, 0.98, 1.09),),
        ),
        Candidate(
            "steady",
            {"ppo.alpha": 2e-4, "ppo.clip": 0.2},
            (_evals(1.00, 1.06, 1.07, 1.06),),
        ),
        Candidate(
            "early-kill",
            {"ppo": {"alpha": 3e-4, "clip": 0.3}},
            (
                _evals(
                    1.08,
                ),
            ),
        ),
        Candidate("never-evaluated", {"ppo.alpha": 4e-4}, ((),)),
    ]
    rows = rank(candidates, keys, last=3)
    assert [row["name"] for row in rows] == ["early-kill", "steady", "spiky"]
    assert rows[1]["score"] == pytest.approx((1.06 + 1.07 + 1.06) / 3)
    assert rows[2]["score"] == pytest.approx((1.10 + 0.98 + 1.09) / 3)
    assert rows[0]["evals"] == 1 and rows[1]["evals"] == 3
    assert rows[0]["ppo.clip"] == 0.3  # nested config read through the dotted key
    assert tsv_columns(keys) == [*FIXED_COLUMNS, "ppo.alpha", "ppo.clip"]
    path = tmp_path / "sweep.tsv"
    write_tsv(path, rows, keys)
    lines = path.read_text().splitlines()
    assert lines[0].split("\t") == tsv_columns(keys)
    assert lines[1].split("\t")[0] == "early-kill"


def test_rank_seeded_means_over_seeds_and_breaks_ties_on_worst_seed() -> None:
    candidates = [
        Candidate(
            "lucky", {}, (_evals(1.09, 1.09), _evals(1.01, 1.01), _evals(1.05, 1.05))
        ),
        Candidate(
            "robust", {}, (_evals(1.05, 1.05), _evals(1.05, 1.05), _evals(1.05, 1.05))
        ),
        Candidate("one-seed", {}, (_evals(1.05, 1.05),)),
    ]
    rows = rank(candidates, [], last=10)
    assert [row["name"] for row in rows] == ["robust", "one-seed", "lucky"]
    assert rows[0]["seeds"] == 3 and rows[1]["seeds"] == 1
    assert rows[2]["score"] == pytest.approx(1.05)
    assert rows[2]["score_min"] == pytest.approx(1.01)


def test_candidate_config_rebuilds_trial_from_base_and_overrides() -> None:
    assert phase_of("empty-a1b2c3d4") == "empty"
    text = candidate_config(
        "empty-a1b2c3d4", {"ppo.alpha": 0.00123, "ppo.hidden_size": 256}, None
    )
    cfg = yaml.safe_load(text)
    assert cfg["training"]["run_name"] == "empty-a1b2c3d4"
    assert cfg["ppo"]["alpha"] == 0.00123
    assert cfg["ppo"]["hidden_size"] == 256
    assert cfg["track"]["opt_time"] > 0  # merged from the map's track.yaml
    fixed = yaml.safe_load(
        candidate_config("empty-x", {"ppo.alpha": 0.00123}, None, {"training.ckpt_interval": 999_000})
    )
    assert fixed["training"]["ckpt_interval"] == 999_000


class _FakeRun:
    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        rows: tuple[dict[str, Any], ...],
        group: str | None = None,
    ) -> None:
        self.name = name
        self.config = config
        self.group = group
        self._rows = rows

    def history(self, keys: list[str], samples: int, pandas: bool) -> list[dict[str, Any]]:
        assert not pandas and samples >= len(self._rows)
        return [{k: row.get(k) for k in keys} for row in self._rows]


class _FakeApi:
    default_entity = "entity"

    def __init__(self, runs: list[_FakeRun], seed_runs: list[_FakeRun] = ()) -> None:
        self._runs = runs
        self._seed_runs = list(seed_runs)
        self.group_queries: list[list[str]] = []

    def sweep(self, path: str) -> Any:
        assert path == "entity/ppo-lab/abc123"
        return SimpleNamespace(
            id="abc123",
            config={"parameters": {"ppo.alpha": {}, "ppo.clip": {}}},
            runs=self._runs,
        )

    def runs(self, path: str, filters: dict[str, Any]) -> list[_FakeRun]:
        assert path == "entity/ppo-lab"
        self.group_queries.append(filters["group"]["$in"])
        return [run for run in self._seed_runs if run.group in filters["group"]["$in"]]


def test_main_writes_tsv_candidates_and_train_commands(
    tmp_path: Any, monkeypatch: Any, capsys: Any
) -> None:
    runs = [
        _FakeRun(
            "empty-aaa", {"ppo.alpha": 1e-4, "ppo.clip": 0.1}, _evals(1.00, 1.01)
        ),
        _FakeRun(
            "empty-bbb", {"ppo.alpha": 2e-4, "ppo.clip": 0.2}, _evals(1.05, 1.06)
        ),
    ]
    # Only the loser of the single-seed ranking went through the seed stage.
    seed_runs = [
        _FakeRun("seed_76", {}, _evals(0.90, 0.92), group="empty-aaa"),
        _FakeRun("seed_693", {}, _evals(0.94, 0.96), group="empty-aaa"),
    ]
    api = _FakeApi(runs, seed_runs)
    monkeypatch.setattr(select_module.wandb, "Api", lambda: api)
    monkeypatch.setattr(select_module, "REPO", tmp_path)
    select_module.main(["abc123", "--top", "1", "--last", "2", "--seeds", "76,693"])
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith(
        "select  sweep_id=abc123  last=2  top=1  seeded=off  seeds=76,693"
    )
    assert (
        (tmp_path / "outputs" / "sweeps" / "abc123.tsv")
        .read_text()
        .splitlines()[1]
        .startswith("empty-bbb\t1\t2\t")
    )
    candidate = tmp_path / "outputs" / "sweeps" / "abc123" / "empty-bbb.yaml"
    assert yaml.safe_load(candidate.read_text())["ppo"]["clip"] == 0.2
    assert not (
        tmp_path / "outputs" / "sweeps" / "abc123" / "empty-aaa.yaml"
    ).exists()
    assert (
        "  ./train outputs/sweeps/abc123/empty-bbb.yaml --seeds 76,693 --wandb --tag seeds-abc123"
        in out
    )
    assert out[-1] == "then: ./select abc123 --seeded --last 2"
    assert api.group_queries == []

    select_module.main(["abc123", "--seeded", "--last", "2"])
    out = capsys.readouterr().out.splitlines()
    assert api.group_queries == [["empty-aaa", "empty-bbb"]]
    # bbb never ran seeds, so it is not ranked; aaa is scored over trial + 2 seeds.
    assert out[-1] == "empty-aaa"
    tsv = (tmp_path / "outputs" / "sweeps" / "abc123.tsv").read_text().splitlines()
    assert len(tsv) == 2 and tsv[1].startswith("empty-aaa\t3\t2\t")


def test_select_carries_trial_warmstart_into_seed_stage(
    tmp_path: Any, monkeypatch: Any, capsys: Any
) -> None:
    runs = [
        _FakeRun(
            "obstacles-aaa",
            {"ppo.alpha": 1e-4, "ppo.hidden_size": 256, "warmstart": "empty-ref"},
            _evals(0.80, 0.82),
        )
    ]
    api = _FakeApi(runs, [])
    monkeypatch.setattr(select_module.wandb, "Api", lambda: api)
    monkeypatch.setattr(select_module, "REPO", tmp_path)
    select_module.main(["abc123", "--top", "1", "--last", "2", "--seeds", "76"])
    out = capsys.readouterr().out.splitlines()
    assert (
        "  ./train outputs/sweeps/abc123/obstacles-aaa.yaml --seeds 76 --wandb "
        "--tag seeds-abc123 --warmstart empty-ref"
    ) in out
    candidate = tmp_path / "outputs" / "sweeps" / "abc123" / "obstacles-aaa.yaml"
    assert yaml.safe_load(candidate.read_text())["ppo"]["hidden_size"] == 256


def test_warmstart_pins_hidden_size_to_source_run(tmp_path: Any, monkeypatch: Any) -> None:
    from ppo import train

    src = tmp_path / "outputs" / "trained_agents" / "empty-ref"
    src.mkdir(parents=True)
    (src / "config.yaml").write_text("ppo:\n  hidden_size: 256\n")
    monkeypatch.setattr(train, "REPO", tmp_path)
    assert train.warmstart_overrides("empty-ref:_final") == {"ppo.hidden_size": 256}
    assert train.warmstart_overrides(None) == {}
    with pytest.raises(RuntimeError, match="no config"):
        train.warmstart_overrides("missing-run")


def test_obstacles_sweep_omits_warmstart_fixed_keys() -> None:
    keys = set(load_sweep_yaml("obstacles")["parameters"])
    assert keys.isdisjoint(
        {"ppo.hidden_size", "ppo.log_std_i", "ppo.caps_lambda_t", "ppo.caps_lambda_s"}
    )
    assert {"ppo.alpha", "ppo.clip", "ppo.gae_lambda", "ppo.n_epochs", "ppo.target_kl"} <= keys


def test_rank_breaks_score_ties_by_steer_rate() -> None:
    def cand(name: str, score: float, steer: float) -> Candidate:
        return Candidate(
            name,
            {},
            (
                [
                    {"_step": 100000 * (i + 1), "score": score, "Eval/SuccessRate": 1.0,
                     "Eval/Pace": score - 1.0, "Eval/SteerRate": steer, "Eval/SpeedCmdRate": 0.05}
                    for i in range(2)
                ],
            ),
        )

    rows = rank(
        [cand("jittery", 1.1025, 0.096), cand("smooth", 1.1019, 0.067), cand("slow", 1.09, 0.01)],
        [],
        2,
    )
    assert [row["name"] for row in rows] == ["smooth", "jittery", "slow"]
