"""Build one ECMWF run into PNG wind and hazard tiles, the manifest and index.html."""

import argparse
import json
import os
import shutil
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from windtiles import hazards, source
from windtiles.encode import encode_tile
from windtiles.grid import LEVELS, ZOOMS, tile_grid, tile_nodes

MANIFEST_URL = "https://pilot-force-tiles.web.app/wind/v1/manifest.json"
STEPS = list(range(6, 67, 3))
MAX_AGE = timedelta(hours=18)
KT_PER_MS = 1.943844
KELVIN = 273.15
HAZARD_LEVELS = ["850", "700", "500", "400", "300", "250", "200", "150", "col"]
STORM_POOL = 4  # max over ±2 nodes = ±0.5°: the model misplaces cells by tens of miles (README «Сверка»)
NEIGHBORS = {  # hPa → (below, above) for the vertical wind shear
    850: (925, 700),
    700: (850, 600),
    500: (600, 400),
    400: (500, 300),
    300: (400, 250),
    250: (300, 200),
    200: (250, 150),
    150: (200, 100),
}

INDEX_HTML = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Pilot Force wind tiles</title>
<p>Wind tiles for Pilot Force. This service is based on data and products of the European Centre
for Medium-Range Weather Forecasts (<a href="https://www.ecmwf.int">ECMWF</a>).
Copyright © {year} ECMWF. Licence <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.
Changed: u, v in knots, T in °C, coarsened to 0.5° and 1°, 8 bits per value, masked below ground.
Derived: turbulence index (Ellrod TI1), icing potential from T and RH, freezing level, thunderstorm index
from MUCAPE and convective precipitation, its zones, MUCAPE and rate widened by 0.5°; turbulence and icing at 0.5° and 1° take
the maximum around each node.
ECMWF does not accept any liability whatsoever for any error or omission in the data, their availability,
or for any loss or damage arising from their use.</p>
</html>
"""


def frames(run: datetime) -> list[datetime]:
    return [run + timedelta(hours=step) for step in STEPS]


def level_fields(fields, level: str, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """u, v (kt) and T (°C) of one level; NaN below ground on pressure levels."""
    if level == "sfc":
        u, v, t = (fields[(name, 0, step)] for name in ("10u", "10v", "2t"))
        return u * KT_PER_MS, v * KT_PER_MS, t - KELVIN
    hpa = int(level)
    u, v, t = (fields[(name, hpa, step)] for name in ("u", "v", "t"))
    ground = fields[("sp", 0, step)] < hpa * 100
    return tuple(np.where(ground, np.nan, x).astype(np.float32) for x in (u * KT_PER_MS, v * KT_PER_MS, t - KELVIN))


def hazard_fields(fields, level: str, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turbulence S, icing S, freezing level FL of a pressure level, NaN below ground;
    thunderstorm S, MUCAPE (J/kg), convective rate (mm/h) of "col", each widened to its maximum within ±0.5°."""
    if level == "col":
        mucape = fields[("mucape", 0, step)]
        rate = hazards.convective_rate(fields[("cp", 0, step)], fields[("cp", 0, step - 3)])
        return tuple(hazards.max_pool(x, STORM_POOL) for x in (hazards.thunderstorm(mucape, rate), mucape, rate))
    hpa = int(level)
    sp = fields[("sp", 0, step)]
    below, above = ((fields[("u", p, step)], fields[("v", p, step)], fields[("gh", p, step)]) for p in NEIGHBORS[hpa])
    turbulence = hazards.turbulence(fields[("u", hpa, step)], fields[("v", hpa, step)], below, above)
    icing = hazards.icing(fields[("t", hpa, step)], fields[("r", hpa, step)])
    temperatures = {p: fields[("t", p, step)] for p in source.PL_LEVELS if p >= 300}
    freezing = hazards.freezing_level(fields[("2t", 0, step)], sp, temperatures)
    ground = sp < hpa * 100
    return tuple(np.where(ground, np.nan, x).astype(np.float32) for x in (turbulence, icing, freezing))


def build(run: datetime, fields, out: Path, levels=LEVELS, hazard_levels=HAZARD_LEVELS, steps=None) -> int:
    """Wind tiles, then hazard tiles of every level and step, then the manifest and index.html; returns the tile count.
    A hazard data failure drops every hazard tile and keeps the wind."""
    steps = STEPS if steps is None else steps
    count = 0
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
        print(f"skip hazards: {error!r}")
    manifest = {"run": _iso(run), "frames": [_iso(run + timedelta(hours=step)) for step in steps]}
    (out / "wind/v1/manifest.json").write_text(json.dumps(manifest))
    (out / "index.html").write_text(INDEX_HTML.format(year=run.year))
    return count


def _write_hazards(run: datetime, fields, out: Path, hazard_levels, steps) -> int:
    """Hazard tiles of every level and step; returns the tile count."""
    count = 0
    for index, step in enumerate(steps):
        for level in hazard_levels:
            channels = hazard_fields(fields, level, step)
            if index == 0:
                _hazard_summary(level, channels[: 1 if level == "col" else 2])
            pooled = 0 if level == "col" else 2  # col is already widened in hazard_fields
            count += _write_tiles(out / f"hazards/v1/{_frame(run, step)}/{level}", channels, pooled)
    return count


def _frame(run: datetime, step: int) -> str:
    return f"{run:%Y%m%d%H}/{run + timedelta(hours=step):%Y%m%d%H}"


def _write_tiles(directory: Path, channels, pooled: int) -> int:
    """Every zoom's tiles of one level; the first `pooled` channels (S) take the maximum around each node."""
    count = 0
    for zoom, stride in ZOOMS.items():
        globes = [hazards.max_pool(x, stride) if i < pooled and stride > 1 else x for i, x in enumerate(channels)]
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
    from ecmwf.opendata import Client

    parser = argparse.ArgumentParser(prog="windtiles.build")
    parser.add_argument("--out", type=Path, required=True, help="site directory")
    parser.add_argument("--work", type=Path, required=True, help="GRIB download directory, outside the repository")
    parser.add_argument("--force", action="store_true", help="build without comparing with the deployed run")
    args = parser.parse_args(argv)

    client = Client(source=source.SOURCE)
    run = source.latest_run(client)
    if not args.force:
        deployed = deployed_run()
        decision = decide(run, deployed, datetime.now(timezone.utc))
        if decision != "build":
            if decision == "stale":
                print(f"deployed run {_iso(deployed)} older than 18 h, no newer run with step 66")
            else:
                print(f"run {_iso(run)} up to date")
            _output(changed=False)
            return 1 if decision == "stale" else 0

    print(f"building run {_iso(run)}")
    fields = source.read_fields(source.download(client, run, STEPS, args.work))
    print(f"{build(run, fields, args.out)} tiles in {args.out}")
    _output(changed=True)
    return 0


def _output(changed: bool) -> None:
    if path := os.environ.get("GITHUB_OUTPUT"):
        with open(path, "a") as file:
            file.write(f"changed={'true' if changed else 'false'}\n")


if __name__ == "__main__":
    raise SystemExit(main())
