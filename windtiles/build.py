"""Build one GFS run into PNG wind and hazard tiles, the manifest and index.html."""

import argparse
import json
import os
import shutil
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from windtiles import hazards, models, source, tiles
from windtiles.encode import encode_tile
from windtiles.grid import LEVELS, ZOOMS, tile_grid, tile_nodes

MANIFEST_URL = "https://ga-tiles.aviationweather.pro/wind/v1/manifest.json"
STEPS = list(models.GFS.steps)
MAX_AGE = timedelta(hours=18)
KELVIN = 273.15
HAZARD_LEVELS = [*LEVELS, "col"]
STORM_POOL = 4  # max over ±2 nodes = ±0.5°: the model misplaces cells by tens of miles (README «Сверка»)
NEIGHBORS = {500: (600, 400), 400: (500, 300), 300: (400, 250)}  # hPa → (below, above) for the vertical wind shear
SKY_POOLS = {"base": ("min", "min", "max"), "cloud": ("mean", "mean", "mean"), "sky": ("mean", "node", "max")}
CLOUD_TOP_PERCENT = 50  # the cloud top is the highest level with at least this cover

INDEX_HTML = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Aviation Weather wind tiles</title>
<p>Wind tiles for Aviation Weather. Based on NOAA GFS 0.25° data, public domain. NOAA does not endorse this service.
Changed: u, v in knots, T in °C, coarsened to 0.5° and 1°, 8 bits per value, masked below ground.
Derived: turbulence index (Ellrod TI1), icing potential from T and RH, freezing level in hundreds of feet,
thunderstorm index from CAPE and convective precipitation rate, its zones, CAPE and rate widened by 0.5°;
turbulence and icing at 0.5° and 1° take the maximum around each node;
ceiling above ground in feet, capped at 12,000, visibility in statute miles, capped at 10, gust in knots;
cloud cover at 0.5° and 1° averaged around each node, boundary layer height in feet,
cloud top in feet from the highest pressure level with at least 50 % cover.</p>
</html>
"""


def frames(run: datetime) -> list[datetime]:
    return [run + timedelta(hours=step) for step in STEPS]


def level_fields(fields, level: str, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """u, v (kt) and T (°C) of one level; NaN below ground on pressure levels."""
    if level == "sfc":
        u, v, t = (fields[(name, 0, step)] for name in ("10u", "10v", "2t"))
        return u * hazards.KT_PER_MS, v * hazards.KT_PER_MS, t - KELVIN
    hpa = int(level)
    u, v, t = (fields[(name, hpa, step)] for name in ("u", "v", "t"))
    ground = fields[("sp", 0, step)] < hpa * 100
    return tuple(np.where(ground, np.nan, x).astype(np.float32) for x in (u * hazards.KT_PER_MS, v * hazards.KT_PER_MS, t - KELVIN))


def hazard_fields(fields, level: str, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turbulence S (0 below 500 hPa), icing S (0 above 600 hPa), freezing level in hundreds of feet above sea level
    of a pressure level, NaN below ground; thunderstorm S, CAPE (J/kg), convective rate (mm/h) of "col", each widened
    to its maximum within ±0.5°; ceiling above ground (ft), visibility (SM), gust (kt) of "sfc"."""
    if level == "sfc":
        ceiling = hazards.ceiling_ft(fields[("ceiling", 0, step)], fields[("orog", 0, step)])
        return ceiling, hazards.visibility_sm(fields[("vis", 0, step)]), hazards.gust_kt(fields[("gust", 0, step)])
    if level == "col":
        cape = fields[("cape", 0, step)]
        rate = hazards.convective_rate(fields[("cprat", 0, step)])
        return tuple(hazards.max_pool(x, STORM_POOL) for x in (hazards.thunderstorm(cape, rate), cape, rate))
    hpa = int(level)
    sp = fields[("sp", 0, step)]
    turbulence = icing = np.zeros_like(sp)
    if hpa in NEIGHBORS:
        below, above = (tuple(fields[(name, p, step)] for name in ("u", "v", "gh")) for p in NEIGHBORS[hpa])
        turbulence = hazards.turbulence(fields[("u", hpa, step)], fields[("v", hpa, step)], below, above)
    if hpa in source.ICING_MB:
        icing = hazards.icing(fields[("t", hpa, step)], fields[("r", hpa, step)])
    freezing = hazards.freezing_fl(fields[("zero", 0, step)], fields[("2t", 0, step)])
    ground = sp < hpa * 100
    return tuple(np.where(ground, np.nan, x).astype(np.float32) for x in (turbulence, icing, freezing))


def sky_fields(fields, folder: str, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`base`: ceiling above ground (ft, 0…12,000), visibility (SM, 0…10), CAPE (J/kg); `cloud`: low, middle and high
    cloud cover (%); `sky`: total cover (%), boundary layer height (ft above ground), cloud top (ft above sea level):
    the highest of `source.CLOUD_MB` above ground with cover ≥ `CLOUD_TOP_PERCENT`, NaN where there is none."""
    get = lambda name, level=0: fields[(name, level, step)]
    if folder == "base":
        return hazards.ceiling_ft(get("ceiling"), get("orog")), hazards.visibility_sm(get("vis")), get("cape")
    if folder == "cloud":
        return get("lcc"), get("mcc"), get("hcc")
    top = np.full_like(get("sp"), np.nan)
    for mb in source.CLOUD_MB:  # from the ground up: a higher cloudy level overwrites
        cloudy = (get("tcc", mb) >= CLOUD_TOP_PERCENT) & (get("sp") >= mb * 100)
        top = np.where(cloudy, get("gh", mb) / hazards.M_PER_FT, top)
    return get("tcc"), get("hpbl") / hazards.M_PER_FT, top


def build_sky(run: datetime, fields, out: Path, steps=None) -> int:
    """GFS sky tiles of every frame and folder, the GFS manifest and models.json; returns the tile count. A missing
    field costs that frame's folder alone."""
    model = models.GFS
    steps = list(model.steps) if steps is None else steps
    count = 0
    for step in steps:
        for folder, pools in SKY_POOLS.items():
            try:
                channels = sky_fields(fields, folder, step)
            except KeyError as error:
                print(f"skip sky {folder} at step {step}: {error!r}")
                continue
            directory = out / f"sky/v1/{model.id}/{_frame(run, step)}/{folder}"
            count += tiles.write_tiles(directory, channels, pools, *model.bbox[:2], model.zoom)
    _write_manifest(out / f"sky/v1/{model.id}/manifest.json", run, steps)
    write_json(out / "sky/v1/models.json", models.models_json(out))
    return count


def build(
    run: datetime, fields, out: Path, levels=LEVELS, hazard_levels=HAZARD_LEVELS, steps=None
) -> tuple[int, bool]:
    """Wind tiles, then hazard tiles of every level and step, then the sky family, then the manifest and index.html;
    returns the tile count and whether the hazards were dropped. A hazard data failure drops every hazard tile and keeps
    the wind; a sky one costs its own tiles alone."""
    steps = STEPS if steps is None else steps
    count = 0
    dropped = False
    for index, step in enumerate(steps):
        for level in levels:
            u, v, t = level_fields(fields, level, step)
            if index == 0:
                _summary(level, u, v, t)
            count += _write_tiles(out / f"wind/v1/{_frame(run, step)}/{level}", (u, v, t), pooled=0)
    try:
        count += _write_hazards(run, fields, out, hazard_levels, steps)
    except (KeyError, ValueError) as error:
        shutil.rmtree(out / "hazards", ignore_errors=True)
        dropped = True
        print(f"skip hazards: {error!r}")
    count += build_sky(run, fields, out, steps)
    _write_manifest(out / "wind/v1/manifest.json", run, steps)
    (out / "index.html").write_text(INDEX_HTML)
    return count, dropped


def _write_manifest(path: Path, run: datetime, steps) -> None:
    write_json(path, json.dumps({"run": _iso(run), "frames": [_iso(run + timedelta(hours=step)) for step in steps]}))


def write_json(path: Path, text: str) -> None:
    """Through a temporary file and a rename: a reader of a live site sees the old file or the new one, never half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _write_hazards(run: datetime, fields, out: Path, hazard_levels, steps) -> int:
    """Hazard tiles of every level and step; returns the tile count. A broken surface field costs that frame's `sfc`
    tiles alone: the conditions are one layer, not all the hazards."""
    count = 0
    for index, step in enumerate(steps):
        for level in hazard_levels:
            directory = out / f"hazards/v1/{_frame(run, step)}/{level}"
            if level == "sfc":
                try:
                    count += _write_tiles(directory, hazard_fields(fields, level, step), pooled=2, pool=hazards.min_pool)
                except KeyError as error:
                    print(f"skip sfc at step {step}: {error!r}")
                continue
            channels = hazard_fields(fields, level, step)
            if index == 0:
                _hazard_summary(level, channels[: 1 if level == "col" else 2])
            pooled = 0 if level == "col" else 2  # col is already widened in hazard_fields
            count += _write_tiles(directory, channels, pooled)
    return count


def _frame(run: datetime, step: int) -> str:
    return f"{run:%Y%m%d%H}/{run + timedelta(hours=step):%Y%m%d%H}"


def _write_tiles(directory: Path, channels, pooled: int, pool=hazards.max_pool) -> int:
    """Every zoom's tiles of one level; the first `pooled` channels take the `pool` around each node: the maximum of S,
    the minimum of ceiling and visibility."""
    count = 0
    for zoom, stride in ZOOMS.items():
        globes = [pool(x, stride) if i < pooled and stride > 1 else x for i, x in enumerate(channels)]
        rows, columns = tile_grid(zoom)
        for row in range(rows):
            for column in range(columns):
                path = directory / f"{zoom}/{row}/{column}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(encode_tile(*(tile_nodes(x, zoom, row, column) for x in globes)))
                count += 1
    return count


def _summary(level: str, u, v, t) -> None:
    speed = np.hypot(u, v)
    print(
        f"{level:>4}: max {np.nanmax(speed):.0f} kt, T {np.nanmin(t):.1f} … {np.nanmax(t):.1f} °C,"
        f" no data {np.isnan(speed).mean():.1%}"
    )


def _hazard_summary(level: str, severities) -> None:
    shares = ", ".join(f"S ≥ 0.25 {np.mean(s >= 0.25):.1%} ≥ 0.5 {np.mean(s >= 0.5):.1%}" for s in severities)
    print(f"{level:>4} hazards: {shares}")


def _iso(time: datetime) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ")


def deployed_run(url: str = MANIFEST_URL) -> datetime | None:
    """Run of the deployed manifest; None on any network or parse error."""
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            run = datetime.fromisoformat(json.load(response)["run"])
        return run if run.tzinfo else run.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def decide(latest: datetime, deployed: datetime | None, now: datetime) -> str:
    if deployed is None or deployed < latest:
        return "build"
    if deployed == latest and now - deployed > MAX_AGE:
        return "stale"
    return "up to date"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="windtiles.build")
    parser.add_argument("--out", type=Path, required=True, help="site directory")
    parser.add_argument("--work", type=Path, required=True, help="GRIB download directory, outside the repository")
    parser.add_argument("--force", action="store_true", help="build without comparing with the deployed run")
    parser.add_argument("--deployed", default=MANIFEST_URL, help="URL of the deployed manifest, file:// in CI")
    parser.add_argument("--sky-only", action="store_true", help="the sky family alone, for a local check")
    args = parser.parse_args(argv)

    run = source.latest_run(datetime.now(timezone.utc), source.exists)
    if run is None:
        print("no GFS run with step 66 in the last 24 hours")
        _output(changed=False)
        return 1
    deployed = deployed_run(args.deployed)
    if not args.force:
        decision = decide(run, deployed, datetime.now(timezone.utc))
        if decision != "build":
            if decision == "stale":
                print(f"deployed run {_iso(deployed)} older than 18 h, no newer run with step 66")
            else:
                print(f"run {_iso(run)} up to date")
            _output(changed=False)
            return 1 if decision == "stale" else 0

    print(f"building run {_iso(run)}")
    if args.sky_only:
        fields = source.read_fields(source.download(run, STEPS, args.work, fields=source.SKY_FIELDS))
        print(f"{build_sky(run, fields, args.out)} sky tiles in {args.out}")
        return 0
    fields = source.read_fields(source.download(run, STEPS, args.work))
    count, dropped = build(run, fields, args.out)
    print(f"{count} tiles in {args.out}")
    if dropped:
        # A wind-only site replaces the deployed hazards: only when these are gone or old anyway. Otherwise the deployed
        # run stays, it is older than the latest, and the next hour retries.
        if deployed is not None and datetime.now(timezone.utc) - deployed <= MAX_AGE:
            print(f"::error::hazards dropped, deployed run {_iso(deployed)} stays")
            _output(changed=False)
            return 1
        print(f"::error::hazards dropped, deploying the wind of run {_iso(run)} alone")
    _output(changed=True)
    return 0


def _output(changed: bool) -> None:
    if path := os.environ.get("GITHUB_OUTPUT"):
        with open(path, "a") as file:
            file.write(f"changed={'true' if changed else 'false'}\n")


if __name__ == "__main__":
    raise SystemExit(main())
