"""Pydantic schema for the keys training reads."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

REPO = Path(__file__).resolve().parents[1]


class PPOConfig(BaseModel):
    """PPO hyperparameters, reward weights, and CAPS smoothness weights."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    batch_size: int = Field(gt=0)
    rollout: int = Field(gt=0)
    clip: float = Field(gt=0.0, le=0.5)
    c2_i: float = Field(ge=0.0, le=1.0)
    c2_f: float = Field(ge=0.0, le=1.0)
    gae_lambda: float = Field(ge=0.0, le=1.0)
    gamma: float = Field(ge=0.0, le=1.0)
    alpha: float = Field(gt=0.0, lt=0.1)
    log_std_i: float = Field(ge=-5.0, le=2.0)
    n_epochs: int = Field(ge=1, le=20)
    normalise_advantages: bool
    target_kl: float = Field(gt=0.0, lt=1.0)
    c1: float = Field(ge=0.0, le=1.0)
    hidden_size: int = Field(default=256, ge=32, le=512)
    r_progress: float = Field(gt=0.0)
    r_collision: float = Field(lt=0.0)
    r_lap: float = Field(gt=0.0)
    caps_lambda_t: float = Field(default=1.0, ge=0.0)
    caps_lambda_s: float = Field(default=0.5, ge=0.0)
    caps_lambda_t_v: float = Field(default=0.25, ge=0.0)
    caps_lambda_s_v: float = Field(default=0.5, ge=0.0)
    caps_sigma_s: float = Field(default=0.05, gt=0.0)
    caps_scale_hold: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("c2_f")
    @classmethod
    def final_c2_should_be_less_than_initial(cls, v: float, info: ValidationInfo) -> float:
        """Reject an entropy coefficient that grows over the run."""
        if "c2_i" in info.data and v > info.data["c2_i"]:
            raise ValueError(f"c2_f ({v}) should be <= c2_i ({info.data['c2_i']})")
        return v


class TrackConfig(BaseModel):
    """Map name from the yaml plus numbers filled from the track file at load."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    map: str
    opt_time: float
    length: float
    direction: str
    start_pose: list[float]
    # Plant speed cap for this map; None = the vehicle's max_speed_mps. Set by
    # `./track --vmax`, which also bounds the minimum-time lap behind opt_time.
    max_speed_mps: Optional[float] = Field(default=None, gt=0.0)
    # Top speed the safety filter certifies a stop from and caps commands at;
    # None = the vehicle's max_speed_mps. Independent of the plant cap: the
    # stop horizon grows linearly with it (sf SpeedLimits).
    filter_v_max_mps: Optional[float] = Field(default=None, gt=0.0)


class TrainingConfig(BaseModel):
    """Run length, checkpoints, evaluation, and plant noise."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    run_name: str
    n_phase1_steps: int = Field(default=0, ge=0)
    n_phase2_steps: int = Field(default=0, ge=0)
    ckpt_interval: int = Field(default=250_000, ge=1000)
    eval_interval: int = Field(default=100_000, ge=1000)
    patience_evals: int = Field(default=40, ge=1)
    eval_set: Optional[str] = None
    n_empty_envs: Optional[int] = Field(default=None, ge=0)
    # When an obstacle env draws a new layout: "complete" keeps it through crashes
    # until num_laps flying laps are done; "episode" redraws on every reset;
    # "never" keeps each env's initial layout for the whole run (one layout per
    # obstacle env).
    layout_switch: Literal["complete", "episode", "never"] = "complete"
    sim2real: bool
    num_laps: int = Field(default=2, ge=1)


class ObstaclesConfig(BaseModel):
    """Counts and sizes used when sampling training layouts."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    n_obstacles: int = Field(ge=0)
    min_s_distance: float = Field(gt=0.0)
    obstacle_width: float = Field(gt=0.0)
    obstacle_length: float = Field(gt=0.0)
    centre_deadzone: float = Field(ge=0.0)
    start_exclusion: float = Field(ge=0.0)
    zones_file: Optional[str] = None


class Config(BaseModel):
    """Root training configuration."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    ppo: PPOConfig
    track: TrackConfig
    training: TrainingConfig
    obstacles: ObstaclesConfig

    @model_validator(mode="after")
    def check_phase1_rollout_alignment(self) -> "Config":
        """Warn when the phase-1 budget does not land on a rollout boundary."""
        if self.training.n_phase2_steps > 0:
            roll = self.ppo.rollout
            phase1 = self.training.n_phase1_steps
            if phase1 > 0 and phase1 % roll != 0:
                warnings.warn(
                    f"n_phase1_steps ({phase1}) must be a multiple of rollout ({roll})."
                )
        return self


def resolve(phase_or_path: str) -> Path:
    """Turn a phase name, file path, or trained-run name into a config path."""
    given = Path(phase_or_path)
    if given.exists():
        return given.resolve()
    named = REPO / "ppo" / "configs" / f"{phase_or_path}.yaml"
    if named.exists():
        return named
    run_cfg = REPO / "outputs" / "trained_agents" / phase_or_path / "config.yaml"
    if run_cfg.exists():
        return run_cfg
    raise FileNotFoundError(
        f"no config for {phase_or_path!r}: tried path, ppo/configs/{phase_or_path}.yaml, "
        f"and outputs/trained_agents/{phase_or_path}/config.yaml"
    )


def _track_yaml(map_name: str) -> dict[str, Any]:
    """Read map-specific numbers from sf/tracks/<map>/track.yaml."""
    path = REPO / "sf" / "tracks" / map_name / "track.yaml"
    if not path.exists():
        raise FileNotFoundError(f"track file not found: {path}")
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise KeyError(f"track file {path} is not a mapping")
    return data


def _set_model(cfg: BaseModel, dotted: str, value: Any) -> None:
    """Assign a dotted field on a config object, raising KeyError if any part is missing."""
    parts = dotted.split(".")
    obj: Any = cfg
    for name in parts[:-1]:
        if not isinstance(obj, BaseModel) or name not in obj.__class__.model_fields:
            raise KeyError(dotted)
        obj = getattr(obj, name)
    leaf = parts[-1]
    if not isinstance(obj, BaseModel) or leaf not in obj.__class__.model_fields:
        raise KeyError(dotted)
    setattr(obj, leaf, value)


def _config_from_raw(raw: dict[str, Any]) -> Config:
    """Build Config and turn unknown fields into KeyError."""
    try:
        return Config(**raw)
    except ValidationError as exc:
        for err in exc.errors():
            if err["type"] == "extra_forbidden":
                loc = ".".join(str(part) for part in err["loc"])
                raise KeyError(loc) from exc
        raise


def _merge_track(raw: dict[str, Any]) -> None:
    """Fill track numbers from the map's track.yaml into the loaded mapping."""
    track = raw.get("track")
    if not isinstance(track, dict) or "map" not in track:
        raise KeyError("track.map")
    map_name = str(track["map"])
    data = _track_yaml(map_name)
    merged = {
        "map": data.get("map", map_name),
        "opt_time": data["opt_time"],
        "length": data["length"],
        "direction": data["direction"],
        "start_pose": data["start_pose"],
    }
    for key in ("max_speed_mps", "filter_v_max_mps"):
        if key in data:
            merged[key] = data[key]
    raw["track"] = merged


def load(path: Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a yaml config, merge the map's track.yaml, then apply dotted overrides."""
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise KeyError(f"config {path} is not a mapping")
    overrides = dict(overrides or {})
    if "track.map" in overrides:
        raw.setdefault("track", {})["map"] = str(overrides.pop("track.map"))
    _merge_track(raw)
    cfg = _config_from_raw(raw)
    for dotted, value in overrides.items():
        _set_model(cfg, dotted, value)
    return cfg


# Seed table: ./train defaults to the first, sweep trials the first, ./select the other two;
# the thesis tables use all three.
SEEDS = [17, 76, 693]

# Sweep objective: success first, pace as the tie-breaker.
SCORE_PACE_WEIGHT = 0.1


def score(success_rate: float, pace: float) -> float:
    """Return the selection score of one evaluation."""
    return float(success_rate) + SCORE_PACE_WEIGHT * float(pace)


def parse_set(items: Sequence[str]) -> dict[str, Any]:
    """Parse repeated --set key=value flags into a dotted override dict."""
    overrides: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"expected key=value, got {item!r}")
        overrides[key] = yaml.safe_load(raw)
    return overrides


def dump(cfg: Config) -> str:
    """Serialize the effective config to yaml."""
    return yaml.safe_dump(cfg.model_dump(), sort_keys=False)
