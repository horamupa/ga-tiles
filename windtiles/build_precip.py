"""Build the precip family of the Open-Meteo bucket models: the precipitation sum of the next 12, 24 and 48 hours of the
latest run of each into precip/v1/{model}/{run}/{start}, then precip/v1/index.json of every model the site holds.

    .venv/bin/python -I -m windtiles.build_precip --out site [--models iconEu rrfs] [--workers 8] [--deployed URL]
    .venv/bin/python -I -m windtiles.build_precip --index DIR   # index.json of the manifests in DIR to stdout
"""

import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from windtiles import mercator, models, om_source
from windtiles.build import _output, write_json

WINDOWS = (12, 24, 48)  # hours
VARIABLE, UNIT = "precipitation", "mm"  # a frame valid at t holds the sum over (t − 1 h, t]


def bucket_models() -> list[models.Model]:
    return [model for model in models.MODELS if model.source != "gfs"]


def iso(time: datetime) -> str:
    return time.strftime("%Y-%m-%dT%H:%MZ")


def window_start(run: datetime, now: datetime) -> datetime:
    """The current UTC hour, but never earlier than the run."""
    return max(now.replace(minute=0, second=0, microsecond=0), run)


def build_model(
    model: models.Model, out: Path, workers: int = 8, latest_run=om_source.latest_run, read=om_source.read,
    deployed: dict | None = None, now: datetime | None = None,
) -> tuple[datetime, datetime, list[str], int, int] | None:
    """Sums of the windows the latest run covers from the start hour, then the manifest of the windows written; returns
    the run, the start, the windows, the tile and downloaded byte counts, None when the `deployed` manifest has this
    run and start. Frames are read `workers` at a time and added to the running sums as they arrive. A frame that fails
    or is not in mm drops every window that covers it; with no window left the manifest stays as it was."""
    run = latest_run(model.source, model.runs)
    start = window_start(run.time, now or datetime.now(timezone.utc))
    if deployed and (deployed.get("run"), deployed.get("start")) == (iso(run.time), iso(start)):
        print(f"{model.id}: run {iso(run.time)}, start {iso(start)} up to date", flush=True)
        return None
    covered = [n for n in WINDOWS if all(start + timedelta(hours=h) in run.valid_times for h in range(1, n + 1))]
    if not covered:
        raise RuntimeError(f"{model.id}: run {iso(run.time)} has no window from {iso(start)}")

    sums: dict[int, np.ndarray] = {}
    read_hours: set[int] = set()
    lock = threading.Lock()
    origin = {}  # south, west of the frames' grid

    def frame(hour: int) -> int:
        try:
            fields = read(model.source, run.time, start + timedelta(hours=hour), [VARIABLE], model.zoom)
            if fields.units[VARIABLE] != UNIT:
                raise ValueError(f"unit {fields.units[VARIABLE]!r}, expected {UNIT!r}")
            values = fields.values[VARIABLE]
            columns = round(360 / 2 ** (1 - model.zoom))
            if fields.west == -180 and values.shape[1] == columns + 1:
                values = values[:, :-1]  # 180°E is column 0 again: write_mercator_tiles then wraps
            with lock:
                for n in (n for n in covered if hour <= n):
                    if n in sums:
                        sums[n] += values
                    else:
                        sums[n] = values.astype(np.float32, copy=True)
                read_hours.add(hour)
                origin.update(south=fields.south, west=fields.west)
            return fields.downloaded
        except Exception as error:
            print(f"{model.id}: skip frame +{hour} h from {iso(start)}: {error!r}", flush=True)
            return 0

    with ThreadPoolExecutor(workers) as pool:
        downloaded = sum(pool.map(frame, range(1, max(covered) + 1)))
    om_source.grid_map.cache_clear()
    complete = [n for n in covered if read_hours >= set(range(1, n + 1))]
    if not complete:
        raise RuntimeError(f"{model.id}: no complete window of run {iso(run.time)} from {iso(start)}")
    step = 2.0 ** (1 - model.zoom)
    count = 0
    for n in complete:
        directory = out / f"precip/v1/{model.id}/{run.time:%Y%m%d%H}/{start:%Y%m%d%H}/{n}h"
        field = sums.pop(n)
        count += mercator.write_mercator_tiles(directory, field, origin["south"], origin["west"], step, model.zoom)
    windows = [f"{n}h" for n in complete]
    manifest = {"run": iso(run.time), "start": iso(start), "windows": windows, "maxZoom": model.zoom}
    write_json(out / f"precip/v1/{model.id}/manifest.json", json.dumps(manifest))
    return run.time, start, windows, count, downloaded


def index_json(site: Path) -> str:
    """What the app reads: the legend, the no-data colour, then, in registry order, every bucket model whose manifest
    is in `site`: id, title, km, bbox and home (if any, never for a global model) from the registry, maxZoom, run,
    start and windows from the manifest."""
    legend = [{"mm": mm, "color": _hex(rgba)} for mm, rgba in mercator.LEGEND]
    listed = []
    for model in bucket_models():
        path = site / f"precip/v1/{model.id}/manifest.json"
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        keys = ("id", "title", "km", "bbox", "home")
        entry = {key: value for key in keys if (value := getattr(model, key)) is not None}
        if model.bbox == models.GLOBE:
            entry.pop("home", None)
        listed.append(entry | {key: manifest[key] for key in ("maxZoom", "run", "start", "windows")})
    return json.dumps({"legend": legend, "noData": _hex(mercator.NO_DATA), "models": listed})


def _hex(rgba: tuple[int, int, int, int]) -> str:
    """RGBA → "#RRGGBBAA"."""
    return "#" + "".join(f"{c:02X}" for c in rgba)


def deployed_manifest(url: str) -> dict | None:
    """The deployed manifest; None on any network or parse error."""
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)
    except Exception:
        return None


def main(argv=None) -> int:
    bucket = [model.id for model in bucket_models()]
    parser = argparse.ArgumentParser(prog="windtiles.build_precip")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", type=Path, help="site directory")
    target.add_argument(
        "--index", type=Path, help="print precip/v1/index.json of the manifests in this directory (CI: the bucket's)"
    )
    parser.add_argument("--models", nargs="+", choices=bucket, default=bucket, help="default: every bucket model")
    parser.add_argument("--workers", type=int, default=8, help="frames read at once")
    parser.add_argument(
        "--deployed", help="URL of the deployed site, file:// in CI: a model whose run and start are there is skipped"
    )
    args = parser.parse_args(argv)
    if args.index:
        print(index_json(args.index))
        return 0

    failed = changed = False
    for model in (model for model in bucket_models() if model.id in args.models):
        clock = time.monotonic()
        deployed = deployed_manifest(f"{args.deployed}/precip/v1/{model.id}/manifest.json") if args.deployed else None
        try:
            result = build_model(model, args.out, args.workers, deployed=deployed)
        except Exception as error:
            print(f"::error::{model.id}: {error!r}", flush=True)
            failed = True
            continue
        if result is None:
            continue
        changed = True
        run, start, windows, count, downloaded = result
        print(
            f"{model.id}: run {iso(run)}, start {iso(start)}, windows {' '.join(windows)}, {count} tiles,"
            f" {downloaded / 1e6:.0f} MB downloaded, {time.monotonic() - clock:.0f} s",
            flush=True,
        )
    write_json(args.out / "precip/v1/index.json", index_json(args.out))
    _output(changed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
