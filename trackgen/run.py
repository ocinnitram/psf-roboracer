"""Build sf/tracks/<map>/ from a raw SLAM occupancy map."""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from typing import Any

if "--no-plot" in sys.argv:
    import matplotlib

    matplotlib.use("Agg")

import pandas as pd
import yaml

from trackgen.paths import REPO, TrackPaths
from trackgen.stages import centreline, mintime, raceline, screening, zones

STAGES = ("centreline", "raceline", "mintime", "zones", "screening", "compile")
# track.yaml keys this pipeline writes; any other key is hand-set and kept.
PIPELINE_KEYS = ("map", "opt_time", "length", "direction", "start_pose", "max_speed_mps")


def _print_config(cfg: dict[str, Any]) -> None:
    """Print the effective settings as the first stdout lines."""
    print("trackgen")
    for key, value in cfg.items():
        print(f"  {key}: {value}")


def _stage_index(name: str) -> int:
    """Return the index of a stage name."""
    if name not in STAGES:
        raise SystemExit(f"unknown stage {name!r}; choose from {', '.join(STAGES)}")
    return STAGES.index(name)


def _write_track_yaml(
    paths: TrackPaths,
    opt_time: float,
    direction: str,
    vmax: float | None,
) -> None:
    """Write track.yaml from the raceline table and the minimum-time lap."""
    if not paths.raceline_csv.exists():
        raise FileNotFoundError(f"missing raceline: {paths.raceline_csv}")
    df = pd.read_csv(paths.raceline_csv)
    length = float(df["s"].iloc[-1])
    start_pose = [
        float(df["x"].iloc[0]),
        float(df["y"].iloc[0]),
        float(df["track_yaw_rad"].iloc[0]),
    ]
    opt_time_s = float(f"{opt_time:.2f}")
    pose = ", ".join(f"{v:.4f}" for v in start_pose)
    text = (
        f"map: {paths.map_name}\n"
        f"opt_time: {opt_time_s}\n"
        f"length: {length}\n"
        f"direction: {direction}\n"
        f"start_pose: [{pose}]\n"
    )
    if vmax is not None:
        text += f"max_speed_mps: {float(vmax)}\n"
    # Keep hand-set keys the pipeline does not own (e.g. filter_v_max_mps).
    for key, value in _existing_doc(paths).items():
        if key not in PIPELINE_KEYS:
            text += yaml.safe_dump({key: value}, default_flow_style=False)
    paths.track_yaml.write_text(text)
    print(f"wrote {paths.track_yaml}")
    print(text.rstrip())


def _compile(paths: TrackPaths) -> None:
    """Compile the occupancy map and raceline into the sim blob."""
    script = REPO / "sim" / "tools" / "compile_track.py"
    cmd = [sys.executable, str(script), paths.map_name, "--no-plot"]
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO))


def _existing_doc(paths: TrackPaths) -> dict[str, Any]:
    """Return the track.yaml mapping already on disk, or an empty one."""
    if paths.track_yaml.exists():
        doc = yaml.safe_load(paths.track_yaml.read_text())
        if isinstance(doc, dict):
            return doc
    return {}


def _existing_direction(paths: TrackPaths, fallback: str) -> str:
    """Return the stored travel direction, or fallback."""
    direction = _existing_doc(paths).get("direction")
    return str(direction) if direction in {"cw", "ccw"} else fallback


def _existing_vmax(paths: TrackPaths) -> float | None:
    """Return the speed cap already stored in track.yaml, if any."""
    vmax = _existing_doc(paths).get("max_speed_mps")
    return None if vmax is None else float(vmax)


def _existing_opt_time(paths: TrackPaths) -> float:
    """Return opt_time already stored in track.yaml."""
    opt_time = _existing_doc(paths).get("opt_time")
    if opt_time is not None:
        return float(opt_time)
    raise SystemExit(f"missing opt_time in {paths.track_yaml}; run the mintime stage")


def run(
    map_name: str,
    direction: str,
    start_s: float | None,
    plot: bool,
    from_stage: str,
    zone_spec: zones.ZoneSpec = 2,
    vmax: float | None = None,
) -> None:
    """Run the track pipeline from from_stage through track.yaml."""
    paths = TrackPaths(map_name)
    start = _stage_index(from_stage)
    stored_vmax = _existing_vmax(paths)
    if vmax is None:
        vmax = stored_vmax
    elif start > _stage_index("mintime") and vmax != stored_vmax:
        raise SystemExit("--vmax changes opt_time; restart from the mintime stage")
    cfg: dict[str, Any] = {
        "map": str(paths.map_dir),
        "direction": direction,
        "start_s": start_s,
        "zones": zone_spec,
        "vmax": vmax,
        "plot": plot,
        "from": from_stage,
        "out": str(paths.track_dir),
    }
    _print_config(cfg)
    if not paths.yaml_path.exists():
        raise SystemExit(f"missing map yaml: {paths.yaml_path}")
    if not paths.occupancy_image_path().exists():
        raise SystemExit(f"missing occupancy image in {paths.map_dir}")

    traced_direction = _existing_direction(paths, "cw")
    opt_time: float | None = None

    if start <= _stage_index("centreline"):
        print("\n[centreline]")
        traced_direction = centreline.run(paths, direction, start_s, plot)
    if start <= _stage_index("raceline"):
        print("\n[raceline]")
        raceline.run(paths, traced_direction, plot)
    if start <= _stage_index("mintime"):
        print("\n[mintime]")
        opt_time = mintime.run(paths, traced_direction, plot, vmax)
    else:
        opt_time = _existing_opt_time(paths)
    if start <= _stage_index("zones"):
        print("\n[zones]")
        zones.run(paths, plot, zone_spec)
    if start <= _stage_index("screening"):
        print("\n[screening]")
        screening.run(paths)
    if start <= _stage_index("compile"):
        print("\n[compile]")
        _compile(paths)

    print("\n[track.yaml]")
    assert opt_time is not None
    _write_track_yaml(paths, opt_time, traced_direction, vmax)


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the pipeline."""
    # A GUI plotting backend swallows Ctrl-C; keep the default handler so it aborts.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    parser = argparse.ArgumentParser(description="Build sf/tracks/<map>/ from a raw SLAM map.")
    parser.add_argument("map", help="track name under sf/tracks/")
    parser.add_argument("--direction", choices=("cw", "ccw"), required=True,
                        help="travel direction of the car around the track")
    parser.add_argument("--start", type=float, default=None, metavar="S", dest="start_s",
                        help="station on the traced centreline that becomes s = 0")
    parser.add_argument("--zones", default="2", metavar="N|a:b,c:d",
                        help="number of obstacle zones to set with the dials, or explicit station ranges in metres")
    parser.add_argument("--vmax", type=float, default=None, metavar="MPS",
                        help="plant speed cap for this map (default: the cap stored in track.yaml, "
                             "else the vehicle's max_speed_mps); "
                             "bounds the minimum-time lap and is written to track.yaml")
    parser.add_argument("--no-plot", action="store_true", help="skip plots and Enter prompts")
    parser.add_argument(
        "--from",
        dest="from_stage",
        default="centreline",
        metavar="STAGE",
        help="restart at this stage: " + ", ".join(STAGES),
    )
    args = parser.parse_args(argv)
    run(
        map_name=args.map,
        direction=args.direction,
        start_s=args.start_s,
        zone_spec=zones.parse_zone_spec(args.zones),
        plot=not args.no_plot,
        from_stage=args.from_stage,
        vmax=args.vmax,
    )


if __name__ == "__main__":
    main()
