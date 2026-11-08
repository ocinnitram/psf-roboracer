"""FilterLog writes the contracted header and round-trips numbers and strings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evaluation.log import COLUMNS, FilterLog, read


def _row(**updates: object) -> dict[str, object]:
    """One complete log row, with optional field overrides."""
    row: dict[str, object] = {name: 0.0 for name in COLUMNS}
    row["outcome"] = "Passthrough"
    row["selection"] = "CertifiedRl"
    row.update(updates)
    return row


def test_write_header_matches_columns(tmp_path: Path) -> None:
    path = tmp_path / "filter_log.csv"
    log = FilterLog(path)
    log.add(_row(step=1, x=1.5, y=-0.25))
    written = log.write()
    assert written == path
    header = path.read_text().splitlines()[0]
    assert tuple(header.split(",")) == COLUMNS


def test_add_rejects_missing_and_extra_keys(tmp_path: Path) -> None:
    log = FilterLog(tmp_path / "log.csv")
    missing = _row()
    missing.pop("x")
    with pytest.raises(KeyError):
        log.add(missing)
    extra = _row()
    extra["not_a_column"] = 1.0
    with pytest.raises(KeyError):
        log.add(extra)


def test_read_round_trips_numbers_and_strings(tmp_path: Path) -> None:
    path = tmp_path / "filter_log.csv"
    log = FilterLog(path)
    log.add(_row(step=0, sim_time=0.0, x=1.25, v=2.5, outcome="Passthrough", selection="PassLeft"))
    log.add(_row(step=1, sim_time=0.067, x=1.30, v=2.6, outcome="Qp", selection="Corridor"))
    log.write()
    data = read(path)
    assert tuple(data) == COLUMNS
    assert data["outcome"].dtype == object
    assert data["selection"].dtype == object
    assert list(data["outcome"]) == ["Passthrough", "Qp"]
    assert list(data["selection"]) == ["PassLeft", "Corridor"]
    np.testing.assert_allclose(np.asarray(data["step"], dtype=float), [0.0, 1.0])
    np.testing.assert_allclose(np.asarray(data["x"], dtype=float), [1.25, 1.30])
    np.testing.assert_allclose(np.asarray(data["v"], dtype=float), [2.5, 2.6])
    np.testing.assert_allclose(np.asarray(data["sim_time"], dtype=float), [0.0, 0.067])
