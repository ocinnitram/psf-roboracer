"""Paths and occupancy IO for one map under sf/tracks/<map>/."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


class TrackPaths:
    """File locations for one named track directory."""

    def __init__(self, map_name: str) -> None:
        self.map_name = map_name
        self.track_dir = REPO / "sf" / "tracks" / map_name
        self.raw_dir = self.track_dir / "raw"
        self.clean_dir = self.track_dir / "clean"
        self.map_dir = self.clean_dir if self.clean_dir.is_dir() else self.raw_dir
        self.yaml_path = self.map_dir / f"{map_name}.yaml"
        self.centreline_csv = self.track_dir / "centreline.csv"
        self.boundary_csv = self.track_dir / "boundary.csv"
        self.raceline_csv = self.track_dir / "raceline.csv"
        self.raceline_mintime_csv = self.track_dir / "raceline_mintime.csv"
        self.zones_json = self.track_dir / "zones.json"
        self.screening_json = self.track_dir / "screening.json"
        self.track_yaml = self.track_dir / "track.yaml"
        self.blob = self.track_dir / f"{map_name}.blob"

    def map_yaml(self) -> dict[str, Any]:
        """Load the occupancy yaml from clean/ when present, else raw/."""
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"missing map yaml: {self.yaml_path}")
        with open(self.yaml_path) as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"invalid map yaml: {self.yaml_path}")
        return cfg

    def occupancy_image_path(self) -> Path:
        """Return the occupancy image named by the yaml from the map directory."""
        cfg = self.map_yaml()
        name = str(cfg.get("image", f"{self.map_name}.pgm"))
        path = self.map_dir / name
        if path.exists():
            return path
        for ext in (".pgm", ".png"):
            candidate = self.map_dir / f"{self.map_name}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"missing occupancy image in {self.map_dir}")

    def load_occupancy(self) -> tuple[np.ndarray, float, list[float]]:
        """Load the grayscale occupancy image, resolution, and origin xy."""
        cfg = self.map_yaml()
        image = np.array(Image.open(self.occupancy_image_path()).convert("L"))
        resolution = float(cfg["resolution"])
        origin = [float(cfg["origin"][0]), float(cfg["origin"][1])]
        return image, resolution, origin

    def occupied_grid(self) -> np.ndarray:
        """Boolean grid of occupied cells, thresholded as the map yaml prescribes."""
        cfg = self.map_yaml()
        image, _res, _origin = self.load_occupancy()
        occupied = (255.0 - image) / 255.0 > float(cfg.get("occupied_thresh", 0.65))
        if int(cfg.get("negate", 0)):
            occupied = np.logical_not(occupied)
        return occupied
