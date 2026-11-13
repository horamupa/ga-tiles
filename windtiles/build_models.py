"""Build the sky family of the Open-Meteo bucket models: the latest run of each at its `Model.runs` hours into
sky/v1/{model}, then sky/v1/models.json of every model the site holds.

    .venv/bin/python -I -m windtiles.build_models --out site [--models iconEu rrfs] [--workers 8] [--deployed URL]
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from windtiles import models, om_source, tiles
from windtiles.build import SKY_POOLS, _frame, _iso, _output, _write_manifest, deployed_run, write_json
from windtiles.hazards import CEILING_CAP_FT, M_PER_FT, visibility_sm

VARIABLES = {  # layer → bucket variable and its unit; fog reads visibility, cloud_top (RRFS only) is not read
    "clouds": ("cloud_cover", "%"),
    "lowClouds": ("cloud_cover_low", "%"),
    "mediumClouds": ("cloud_cover_mid", "%"),
    "highClouds": ("cloud_cover_high", "%"),
    "cloudBase": ("cloud_base", "m"),
    "visibility": ("visibility", "m"),
    "cape": ("cape", "J/kg"),
    "thermals": ("boundary_layer_height", "m"),
}
GROUND_CLOUD_PERCENT = 50  # UKV cloud_base 0 m: cloud on the ground where the cell's low cover reaches this, else none
VISIBILITY_TOP_M = {"aladin": 7_400}  # ALADIN stops at 7,480 m: from here on it is unlimited, 10 SM


def variables(model: models.Model) -> dict[str, str]:
    """Bucket variable → unit of the model's layers."""
    return dict(VARIABLES[layer] for layer in model.layers if layer in VARIABLES)


def steps(model: models.Model, run: om_source.Run) -> list[int]:
    """The model's frame hours that the run has."""
    return [step for step in model.steps if run.time + timedelta(hours=step) in run.valid_times]


def clean_source(source: dict[str, np.ndarray]) -> None:
    """On the model's own grid, before regridding. A `cloud_base` of 0 m under less than `GROUND_CLOUD_PERCENT` low cover
    means no cloud and becomes +inf, which `om_source.regrid` keeps out of the blend; 0 m under more is cloud on the
    ground (fog) and stays 0. A negative CAPE is no data (HRDPS writes -1000 in a few cells)."""
    if "cloud_base" in source:
        base, low = source["cloud_base"], source["cloud_cover_low"]
        source["cloud_base"] = np.where((base == 0) & (low < GROUND_CLOUD_PERCENT), np.inf, base)
    if "cape" in source:
        source["cape"] = np.where(source["cape"] < 0, np.nan, source["cape"])


def sky_channels(model: models.Model, values: dict[str, np.ndarray]) -> dict[str, tuple]:
    """Folder → R, G, B in the units of `build.sky_fields`, from a frame's bucket variables; a variable the model lacks
    is NaN. Ceiling: `cloud_base` above ground, m → ft; none (+inf) and above the cap → `CEILING_CAP_FT`."""
    nan = np.full(next(iter(values.values())).shape, np.nan, np.float32)
    get = lambda name: values.get(name, nan)
    ceiling = np.clip(get("cloud_base") / M_PER_FT, 0, CEILING_CAP_FT)
    visibility = get("visibility")
    if top := VISIBILITY_TOP_M.get(model.id):
        visibility = np.where(visibility >= top, np.inf, visibility)
    return {
        "base": (ceiling, visibility_sm(visibility), get("cape")),
        "cloud": (get("cloud_cover_low"), get("cloud_cover_mid"), get("cloud_cover_high")),
        "sky": (get("cloud_cover"), get("boundary_layer_height") / M_PER_FT, nan),
    }


def build_frame(model: models.Model, run: datetime, step: int, out: Path, read=om_source.read) -> tuple[int, int]:
    """Tiles of one frame, all three folders; returns the tile count and the bytes downloaded."""
    names = variables(model)
    fields = read(model.source, run, run + timedelta(hours=step), list(names), model.zoom, prepare=clean_source)
    if wrong := {name: unit for name, unit in fields.units.items() if unit != names[name]}:
        raise ValueError(f"units {wrong}, expected {names}")
    values = fields.values
    columns = round(360 / 2 ** (1 - model.zoom))
    if fields.west == -180 and next(iter(values.values())).shape[1] == columns + 1:
        values = {name: x[:, :-1] for name, x in values.items()}  # 180°E is column 0 again: write_tiles then wraps
    count = 0
    for folder, channels in sky_channels(model, values).items():
        if not all(np.isnan(x).all() for x in channels):
            directory = out / f"sky/v1/{model.id}/{_frame(run, step)}/{folder}"
            count += tiles.write_tiles(directory, channels, SKY_POOLS[folder], fields.south, fields.west, model.zoom)
    return count, fields.downloaded


def build_model(
    model: models.Model, out: Path, workers: int = 8, latest_run=om_source.latest_run, read=om_source.read,
    deployed: datetime | None = None,
) -> tuple[datetime, int, int, int] | None:
    """The latest run's frames, `workers` at a time, one frame per worker in memory, then the manifest of the frames
    written; returns the run and the frame, tile and downloaded byte counts, None when the run is the `deployed` one.
    A failed frame is left out of the manifest and its tiles, if any, stay unreferenced: deleting them could break a
    manifest of the same run already on the site. With no frame written the manifest stays as it was."""
    run = latest_run(model.source, model.runs)
    if run.time == deployed:
        print(f"{model.id}: run {_iso(run.time)} up to date", flush=True)
        return None

    def frame(step: int) -> tuple[int, int] | None:
        try:
            return build_frame(model, run.time, step, out, read)
        except Exception as error:
            print(f"{model.id}: skip frame +{step} h: {error!r}", flush=True)
            return None

    hours = steps(model, run)
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(frame, hours))
    om_source.grid_map.cache_clear()
    written = [(step, result) for step, result in zip(hours, results) if result is not None]
    if not written:
        raise RuntimeError(f"{model.id}: no frame of run {_iso(run.time)} written")
    _write_manifest(out / f"sky/v1/{model.id}/manifest.json", run.time, [step for step, _ in written])
    return run.time, len(written), sum(r[0] for _, r in written), sum(r[1] for _, r in written)


def main(argv=None) -> int:
    bucket = [model.id for model in models.MODELS if model.source != "gfs"]
    parser = argparse.ArgumentParser(prog="windtiles.build_models")
    parser.add_argument("--out", type=Path, required=True, help="site directory")
    parser.add_argument("--models", nargs="+", choices=bucket, default=bucket, help="default: every bucket model")
    parser.add_argument("--workers", type=int, default=8, help="frames read and cut at once")
    parser.add_argument(
        "--deployed", help="URL of the deployed site, file:// in CI: a model whose latest run is there is skipped"
    )
    args = parser.parse_args(argv)

    failed = changed = False
    for model in (model for model in models.MODELS if model.id in args.models):
        start = time.monotonic()
        deployed = deployed_run(f"{args.deployed}/sky/v1/{model.id}/manifest.json") if args.deployed else None
        try:
            result = build_model(model, args.out, args.workers, deployed=deployed)
        except Exception as error:
            print(f"::error::{model.id}: {error!r}", flush=True)
            failed = True
            continue
        if result is None:
            continue
        changed = True
        run, frames, count, downloaded = result
        print(
            f"{model.id}: run {_iso(run)}, {frames} frames, {count} tiles, {downloaded / 1e6:.0f} MB downloaded,"
            f" {time.monotonic() - start:.0f} s",
            flush=True,
        )
    write_json(args.out / "sky/v1/models.json", models.models_json(args.out))
    _output(changed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
