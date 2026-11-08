"""Write the obstacle-placement screening file from the measured-sight zones."""
from __future__ import annotations

import json
from typing import Any

from trackgen.paths import TrackPaths

# Screening on this map is a data file: valid_zones_s come from zones.json,
# and the remaining keys are the placement defaults. There is no numeric
# screen; s then n are sampled uniformly inside the zones.
_DESCRIPTION = (
    "Placement config for the obs-set-dev-500 and obs-set-tune-50 evaluation sets. "
    "Stations come from the "
    "measured-sight zones, which already guarantee a passable side, so no "
    "numeric screen and no class composition are applied: s then n are "
    "sampled uniformly."
)

_SCREENING_DEFAULTS: dict[str, Any] = {
    "s_step_m": 0.05,
    "n_step_m": 0.025,
    "conditioning_approach_m": 0.75,
    "clearance_after_m": 0.5,
    "max_abs_curvature_per_m": None,
    "min_track_width_m": None,
    "max_width_variation_m": None,
    "extra_reserve_m": None,
    "boundary_buffer_m": 0.165,
    "collision_margin_m": 0.05,
    "passing_margin_m": 0.02,
    "ego_width_m": 0.29,
    "ego_length_m": 0.5,
    "obstacle_width_m": 0.25,
    "obstacle_length_m": 0.35,
    "visibility_approach_samples_m": [],
}


def run(paths: TrackPaths) -> dict[str, Any]:
    """Write screening.json, copying valid_zones_s from zones.json."""
    if not paths.zones_json.exists():
        raise FileNotFoundError(f"missing zones: {paths.zones_json}")
    zones_doc = json.loads(paths.zones_json.read_text())
    valid = zones_doc.get("valid_zones_s")
    if not isinstance(valid, list):
        raise ValueError(f"zones.json missing valid_zones_s: {paths.zones_json}")
    screening = dict(_SCREENING_DEFAULTS)
    screening["valid_zones_s"] = valid
    doc: dict[str, Any] = {
        "track": paths.map_name,
        "description": _DESCRIPTION,
        "screening": screening,
        "main_mix": None,
    }
    paths.screening_json.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {paths.screening_json}")
    return doc
