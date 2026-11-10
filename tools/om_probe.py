"""Read one frame (+6 h) of each Open-Meteo bucket model and print what it costs.

    .venv/bin/python -I tools/om_probe.py [model …]
"""

import sys
import time
import urllib.request
from datetime import timedelta

import numpy as np

from windtiles.models import MODELS
from windtiles.om_source import file_url, latest_run, read

ZOOMS = {model.source: model.zoom for model in MODELS if model.source != "gfs"}  # maximum zoom per model
WANTED = ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
          "visibility", "cape", "boundary_layer_height", "cloud_base"]
MB = 1e6


def timed(*args):
    start = time.perf_counter()
    fields = read(*args)
    return fields, time.perf_counter() - start


def probe(model: str) -> None:
    zoom = ZOOMS[model]
    run = latest_run(model, hours=(0, 6, 12, 18))
    valid = run.time + timedelta(hours=6)
    names = [name for name in WANTED if name in run.variables]
    leads = [(t - run.time) / timedelta(hours=1) for t in run.valid_times]
    request = urllib.request.Request(file_url(model, run.time, valid), method="HEAD")
    size = int(urllib.request.urlopen(request, timeout=60).headers["Content-Length"])
    print(f"{model}  run {run.time:%Y-%m-%d %HZ}, {len(leads)} frames {leads[0]:.0f}…{leads[-1]:.0f} h, "
          f"completed +{(run.modified - run.time) / timedelta(hours=1):.1f} h, valid +6 h, zoom {zoom}")
    cold, cold_s = timed(model, run.time, valid, names, zoom)
    warm, warm_s = timed(model, run.time, valid, names, zoom)
    print(f"  frame: {len(names)} variables, {warm.downloaded / MB:.1f} MB of a {size / MB:.1f} MB file, "
          f"{cold_s:.1f} s with index map, {warm_s:.1f} s without; south {warm.south}, west {warm.west}")
    for name in names:
        values = cold.values[name]
        one, seconds = timed(model, run.time, valid, [name], zoom)
        print(f"  {name:22} {one.units[name]:5} {str(values.shape):13} NaN {np.isnan(values).mean():6.1%}  "
              f"min {np.nanmin(values):9.1f}  mean {np.nanmean(values):9.1f}  max {np.nanmax(values):9.1f}  "
              f"{one.downloaded / MB:5.2f} MB {seconds:5.1f} s")


if __name__ == "__main__":
    for model in sys.argv[1:] or ZOOMS:
        probe(model)
