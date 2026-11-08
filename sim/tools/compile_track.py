#!/usr/bin/env python3
"""Compile sf/tracks/<map>/ into a memory-mapped track blob.

Usage:
    uv run --frozen python sim/tools/compile_track.py <map>
    uv run --frozen python sim/tools/compile_track.py <map> --no-plot

Reads the map yaml (which names the occupancy image) and image from
`sf/tracks/<map>/clean/` when that directory exists, else `raw/`, plus
`sf/tracks/<map>/raceline.csv`. Writes
`sf/tracks/<map>/<map>.blob` and, unless `--no-plot`, `<map>_overlay.png`.

Blob, little-endian, 64-byte header then packed arrays. The C reader checks
magic `RRBC` and uses the explicit offsets; it does not take sizeof of a struct.

  offset  0  uint32  magic         bytes 'RRBC'
  offset  4  uint32  version       1
  offset  8  int32   W             grid width in cells
  offset 12  int32   H             grid height in cells
  offset 16  float32 resolution    metres per cell
  offset 20  float32 origin_x      world x of cell (0, 0)
  offset 24  float32 origin_y      world y of cell (0, 0)
  offset 28  int32   n_rline       raceline rows
  offset 32  int32   edt_offset    byte offset of the EDT (always 64)
  offset 36  int32   rline_offset  byte offset of the raceline
  offset 40  zero pad to 64 bytes so the float32 arrays are aligned

  EDT:   float32[H * W]         row-major, world-y-up, signed metres
  rline: float32[n_rline * 10]  x, y, s, s_norm, track_yaw_rad, kappa, w_l, w_r, n_x, n_y

The occupancy image stores row 0 at the top. This program flips once so the
baked EDT is world-y-up and the C transform is flip-free:

    gx = (wx - origin_x) / resolution
    gy = (wy - origin_y) / resolution

Unknown trinary cells are walls. The signed field is
`edt(free) - edt(occupied)`, in metres, float32.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

MAGIC = struct.unpack("<I", b"RRBC")[0]
VERSION = 1
HEADER_BYTES = 64
HEADER_FMT = "<2I 2i f 2f i 2i"

RLINE_COLS = [
    "x", "y", "s", "s_norm", "track_yaw_rad",
    "kappa", "w_l", "w_r", "n_x", "n_y",
]
N_RLINE_COLS = len(RLINE_COLS)

REPO = Path(__file__).resolve().parents[2]

_VIRIDIS = np.array(
    [
        [0.267, 0.005, 0.329],
        [0.230, 0.322, 0.545],
        [0.127, 0.567, 0.551],
        [0.369, 0.788, 0.384],
        [0.993, 0.906, 0.144],
    ],
    dtype=np.float64,
)


def _require(path: Path, label: str) -> Path:
    """Exit with one line if the path is missing."""
    if not path.exists():
        sys.exit(f"missing {label}: {path}")
    return path


def load_map_yaml(yaml_path: Path) -> dict:
    """Load the occupancy yaml and return its dict."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        sys.exit(f"invalid map yaml: {yaml_path}")
    return cfg


def load_occupancy(image_path: Path, cfg: dict) -> np.ndarray:
    """Load the occupancy image into a world-y-up free mask, treating unknown as wall."""
    try:
        img = np.array(Image.open(image_path).convert("L"), dtype=np.float64)
    except OSError:
        sys.exit(f"missing occupancy image: {image_path}")
    negate = int(cfg.get("negate", 0))
    p = img / 255.0 if negate else (255.0 - img) / 255.0
    try:
        free_th = float(cfg["free_thresh"])
    except (KeyError, TypeError, ValueError):
        sys.exit("invalid map yaml: missing free_thresh")
    free_mask = (p < free_th).astype(np.uint8)
    return np.flipud(free_mask)


def build_signed_edt(free_mask: np.ndarray, resolution: float) -> np.ndarray:
    """Build the signed Euclidean distance field in metres as float32."""
    dist_to_wall = np.asarray(distance_transform_edt(free_mask), dtype=np.float64)
    dist_to_free = np.asarray(distance_transform_edt(1 - free_mask), dtype=np.float64)
    signed = (dist_to_wall - dist_to_free) * resolution
    return signed.astype(np.float32)


def load_raceline(csv_path: Path) -> np.ndarray:
    """Load the 10-column raceline and assert its header."""
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
    if header != RLINE_COLS:
        sys.exit(f"raceline header mismatch in {csv_path}: {header}")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != N_RLINE_COLS:
        sys.exit(f"raceline shape {data.shape}, expected (N, {N_RLINE_COLS})")
    return data


def pack_blob(
    out_path: Path,
    edt: np.ndarray,
    rline: np.ndarray,
    resolution: float,
    origin,
) -> None:
    """Write the 64-byte header, signed EDT, and raceline into one blob."""
    H, W = edt.shape
    n_rline = rline.shape[0]
    edt_offset = HEADER_BYTES
    rline_offset = edt_offset + edt.size * 4
    header = struct.pack(
        HEADER_FMT,
        MAGIC,
        VERSION,
        W,
        H,
        float(resolution),
        float(origin[0]),
        float(origin[1]),
        n_rline,
        edt_offset,
        rline_offset,
    )
    header += b"\x00" * (HEADER_BYTES - len(header))
    assert len(header) == HEADER_BYTES
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(np.ascontiguousarray(edt, dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(rline, dtype="<f4").tobytes())


def overlay_plot(
    out_png: Path,
    edt: np.ndarray,
    rline: np.ndarray,
    resolution: float,
    origin,
) -> None:
    """Write a PNG of the signed EDT with the raceline drawn on it."""
    H, _ = edt.shape
    lo = float(edt.min())
    hi = float(edt.max())
    t = np.zeros_like(edt, dtype=np.float64) if hi == lo else (edt - lo) / (hi - lo)
    idx = t * (_VIRIDIS.shape[0] - 1)
    i0 = np.clip(np.floor(idx).astype(np.int32), 0, _VIRIDIS.shape[0] - 1)
    i1 = np.clip(i0 + 1, 0, _VIRIDIS.shape[0] - 1)
    f = (idx - i0)[..., None]
    rgb = (1.0 - f) * _VIRIDIS[i0] + f * _VIRIDIS[i1]
    pix = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(np.flipud(pix), mode="RGB")
    draw = ImageDraw.Draw(img)
    gx = (rline[:, 0] - origin[0]) / resolution
    gy = (rline[:, 1] - origin[1]) / resolution
    pts = [(float(x), float((H - 1) - y)) for x, y in zip(gx, gy)]
    if len(pts) >= 2:
        draw.line(pts, fill=(220, 40, 40), width=2)
    if pts:
        x0, y0 = pts[0]
        draw.ellipse((x0 - 3, y0 - 3, x0 + 3, y0 + 3), fill=(255, 255, 255))
    img.save(out_png)


def compile_map(map_name: str, plot: bool) -> Path:
    """Compile one track directory into a blob and optional overlay PNG."""
    track_dir = REPO / "sf" / "tracks" / map_name
    _require(track_dir, "track directory")
    map_dir = track_dir / "clean" if (track_dir / "clean").is_dir() else track_dir / "raw"
    yaml_path = _require(map_dir / f"{map_name}.yaml", "map yaml")
    csv_path = _require(track_dir / "raceline.csv", "raceline")
    cfg = load_map_yaml(yaml_path)
    try:
        resolution = float(cfg["resolution"])
        origin = cfg["origin"][:2]
        image_name = cfg["image"]
    except (KeyError, TypeError, ValueError, IndexError):
        sys.exit(f"invalid map yaml: {yaml_path}")
    image_path = _require(map_dir / image_name, "occupancy image")

    free_mask = load_occupancy(image_path, cfg)
    edt = build_signed_edt(free_mask, resolution)
    rline = load_raceline(csv_path)

    out_blob = track_dir / f"{map_name}.blob"
    pack_blob(out_blob, edt, rline, resolution, origin)
    if plot:
        overlay_plot(track_dir / f"{map_name}_overlay.png", edt, rline, resolution, origin)
    print(
        f"{map_name}: {edt.shape[1]}x{edt.shape[0]} cells, "
        f"{rline.shape[0]} raceline pts -> {out_blob}"
    )
    return out_blob


def main() -> None:
    """Parse arguments and compile the named map."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("map", help="track name under sf/tracks/")
    ap.add_argument("--no-plot", action="store_true", help="skip the overlay PNG")
    args = ap.parse_args()
    compile_map(args.map, plot=not args.no_plot)


if __name__ == "__main__":
    main()
