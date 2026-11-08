from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

VEHICLE_DIR = Path(__file__).resolve().parents[1]
HEADER = VEHICLE_DIR / "params.h"
YAML_PATH = VEHICLE_DIR / "params.yaml"
GEN = VEHICLE_DIR / "gen_params.py"


def _run_gen() -> None:
    spec = importlib.util.spec_from_file_location("gen_params", GEN)
    assert spec is not None and spec.loader is not None, f"cannot load {GEN}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.generate(HEADER, YAML_PATH)


def test_known_car_params() -> None:
    assert HEADER.is_file(), f"missing {HEADER}"
    _run_gen()
    data = yaml.safe_load(YAML_PATH.read_text())
    assert data["mass_kg"] == 3.74
    assert data["max_speed_mps"] == 4.0
    assert data["control_period_s"] == 0.067


def test_generator_reproduces_yaml_exactly() -> None:
    assert HEADER.is_file(), f"missing {HEADER}"
    _run_gen()
    first = YAML_PATH.read_bytes()
    _run_gen()
    assert YAML_PATH.read_bytes() == first
