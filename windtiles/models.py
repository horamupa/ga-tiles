"""Forecast models of the sky family, one record each; `sky/v1/models.json` is written from them."""

import json
from pathlib import Path
from typing import NamedTuple


class Model(NamedTuple):
    id: str  # folder under sky/v1
    title: str  # legend caption
    km: float  # grid spacing
    bbox: tuple[float, float, float, float]  # south, west, north, east of the model's data
    zoom: int  # finest tile zoom, 1…7: nodes 2^(1 − zoom)° apart
    layers: tuple[str, ...]  # MapLayer.rawValue of the app's layers the model has
    steps: tuple[int, ...]  # forecast hours of the frames; a run that ends earlier stops there
    source: str  # "gfs": NOAA GRIB; otherwise the model's id in the Open-Meteo bucket
    runs: tuple[int, ...] = (0, 6, 12, 18)  # UTC hours of the bucket runs to build from


CLOUD_LAYERS = (
    "clouds", "lowClouds", "mediumClouds", "highClouds", "cloudBase", "cloudTops", "visibility", "fog", "cape",
    "thermals",
)
GLOBE = (-90, -180, 90, 180)
GLOBAL_STEPS = tuple(range(3, 67, 3))
REGIONAL_STEPS = tuple(range(1, 49))


def _without(*missing: str) -> tuple[str, ...]:
    return tuple(layer for layer in CLOUD_LAYERS if layer not in missing)


# bbox of a projected or rotated grid (ALADIN, UKV, HRRR, HRDPS) is the latitude/longitude extent of its edges, not the two corner points
# of the bucket's WKT; all rounded outwards to 0.01° (measured 10.11.2026).
GFS = Model("gfs", "GFS", 25, GLOBE, 3, CLOUD_LAYERS, GLOBAL_STEPS, "gfs")
MODELS = [
    GFS,
    Model("ecmwf", "ECMWF", 9, GLOBE, 5, _without("cloudBase", "cloudTops"), GLOBAL_STEPS, "ecmwf_ifs"),
    Model(
        "icon", "ICON", 13, GLOBE, 4, _without("cloudBase", "cloudTops", "visibility", "fog", "thermals"),
        GLOBAL_STEPS, "dwd_icon",
    ),
    Model(
        "iconEu", "ICON-EU", 7, (29.5, -23.5, 70.5, 62.5), 5, _without("cloudBase", "cloudTops", "thermals"),
        REGIONAL_STEPS, "dwd_icon_eu",
    ),
    Model(
        "iconD2", "ICON-D2", 2.2, (43.18, -3.94, 58.08, 20.34), 7, _without("cloudBase", "cloudTops", "thermals"),
        REGIONAL_STEPS, "dwd_icon_d2",
    ),
    Model(
        "aladin", "ALADIN", 2.3, (38.59, -4.65, 57.21, 34.33), 7, _without("cloudBase", "cloudTops", "thermals"),
        REGIONAL_STEPS, "chmi_aladin_central_europe_2km",
    ),
    Model(
        "arome", "AROME", 2.5, (37.5, -12, 55.4, 16), 6,
        _without("cloudBase", "cloudTops", "visibility", "fog", "thermals"), REGIONAL_STEPS,
        "meteofrance_arome_france0025",
    ),
    Model(
        "aromeHd", "AROME-HD", 1.3, (37.5, -12, 55.4, 16), 7,
        _without("clouds", "cloudBase", "cloudTops", "visibility", "fog", "thermals"), REGIONAL_STEPS,
        "meteofrance_arome_france_hd",
    ),
    Model(
        "ukv", "UKV", 2, (44.5, -24.6, 63.02, 15.36), 7, _without("cloudTops", "thermals"), REGIONAL_STEPS,
        "ukmo_uk_deterministic_2km",
    ),
    Model(
        "hrrr", "HRRR", 3, (21.13, -134.1, 52.62, -60.91), 6, _without("cloudBase", "cloudTops"), REGIONAL_STEPS,
        "ncep_hrrr_conus",
    ),
    Model(
        "hrdps", "HRDPS", 2.5, (27.28, -152.74, 70.62, -40.7), 6, ("clouds", "cape"), REGIONAL_STEPS, "cmc_gem_hrdps",
    ),
    Model(  # MSM runs every 3 h, but only 00 and 12Z reach past +39 h (to +78 h)
        "msm", "MSM", 5, (22.4, 120, 47.6, 150), 5, ("clouds", "lowClouds", "mediumClouds", "highClouds"), REGIONAL_STEPS,
        "jma_msm", (0, 12),
    ),
]


def models_json(site: Path) -> str:
    """What the app reads: id, title, km, bbox, zoom and layers of every model whose manifest is in `site`: the list
    follows what the site holds, so a GFS-only build lists GFS alone."""
    keys = ("id", "title", "km", "bbox", "zoom", "layers")
    present = [model for model in MODELS if (site / f"sky/v1/{model.id}/manifest.json").exists()]
    return json.dumps([{key: getattr(model, key) for key in keys} for model in present])
