"""NOAA GFS 0.25°: latest run, download by byte ranges, GRIB2 → normalized globes."""

import urllib.request
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import eccodes
import numpy as np

from windtiles.grid import LEVELS, NI, NJ, STEP_DEG, normalize

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
WIND_MB = [int(level) for level in LEVELS if level != "sfc"]
ICING_MB = [900, 850, 800, 750, 700, 600]
TURBULENCE_MB = [600, 500, 400, 300, 250]
CLOUD_MB = [1000, 975, 950, 925, 900, *range(850, 99, -50)]  # the cloud-top profile
FIELDS = {  # (name, level) of the .idx entries the wind and hazard tiles read; a step without one stops the run
    *((name, f"{mb} mb") for name in ("UGRD", "VGRD", "TMP") for mb in WIND_MB),
    *(("RH", f"{mb} mb") for mb in ICING_MB),
    *((name, f"{mb} mb") for name in ("HGT", "UGRD", "VGRD") for mb in TURBULENCE_MB),
    ("UGRD", "10 m above ground"),
    ("VGRD", "10 m above ground"),
    ("TMP", "2 m above ground"),
    ("PRES", "surface"),
    ("HGT", "0C isotherm"),
    ("CAPE", "180-0 mb above ground"),
    ("CPRAT", "surface"),
    ("HGT", "cloud ceiling"),
    ("HGT", "surface"),
    ("VIS", "surface"),
    ("GUST", "surface"),
}
SKY_FIELDS = {  # the sky family; a step without one of them loses the sky tiles that read it
    ("HGT", "cloud ceiling"),
    ("HGT", "surface"),
    ("VIS", "surface"),
    ("CAPE", "180-0 mb above ground"),
    ("PRES", "surface"),
    ("TCDC", "entire atmosphere"),
    ("LCDC", "low cloud layer"),
    ("MCDC", "middle cloud layer"),
    ("HCDC", "high cloud layer"),
    ("HPBL", "surface"),
    *((name, f"{mb} mb") for name in ("TCDC", "HGT") for mb in CLOUD_MB),
}
NAMES = {  # (shortName, typeOfLevel) → read_fields name
    ("gh", "isothermZero"): "zero",
    ("cpr", "surface"): "cprat",
    ("gh", "cloudCeiling"): "ceiling",
    ("unknown", "surface"): "hpbl",  # HPBL, NCEP local 0.3.196: eccodes has no name; the one such field we fetch
}
SFC_WIND = ["10u", "10v", "2t", "sp"]  # surface fields the wind tiles read
PL_WIND = ["u", "v", "t"]  # pressure-level fields the wind tiles read on LEVELS
WINDS = {"u", "v", "10u", "10v"}
WIND_LIMIT = 150  # m/s; |wind| at or past it is a broken field
RANGES = {  # K, K, Pa, m, %, m, J/kg, kg/m²/s, m, m, m, m/s, %, %, %, %, m
    "t": (150, 350),
    "2t": (150, 350),
    "sp": (30_000, 115_000),
    "gh": (-1_000, 25_000),
    "r": (-10, 160),
    "zero": (-1_000, 10_000),
    "cape": (-1, 20_000),
    "cprat": (-1e-6, 0.1),
    "ceiling": (-1_000, 25_000),
    "orog": (-1_000, 9_000),
    "vis": (0, 100_000),
    "gust": (0, 150),
    "tcc": (-1, 101),
    "lcc": (-1, 101),
    "mcc": (-1, 101),
    "hcc": (-1, 101),
    "hpbl": (-1, 10_000),
}


class IdxEntry(NamedTuple):
    name: str
    level: str
    forecast: str
    start: int
    end: int | None  # last byte; None for the last entry of a file


def file_url(run: datetime, step: int) -> str:
    return f"{BUCKET}/gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.0p25.f{step:03}"


def latest_run(now: datetime, exists: Callable[[str], bool]) -> datetime | None:
    """Newest run of the last 24 hours that already has step 66."""
    boundary = now.replace(hour=now.hour // 6 * 6, minute=0, second=0, microsecond=0)
    for hours in range(0, 25, 6):
        run = boundary - timedelta(hours=hours)
        if exists(file_url(run, 66) + ".idx"):
            return run
    return None


def parse_idx(text: str) -> list[IdxEntry]:
    rows = [line.split(":") for line in text.splitlines()]
    starts = [int(row[1]) for row in rows]
    ends = [start - 1 for start in starts[1:]] + [None]
    return [IdxEntry(row[3], row[4], row[5], start, end) for row, start, end in zip(rows, starts, ends)]


def wanted(entry: IdxEntry, step: int, fields=FIELDS | SKY_FIELDS) -> bool:
    """One of `fields` at this step; `0-N hour ave fcst` averages do not pass."""
    return (entry.name, entry.level) in fields and entry.forecast == f"{step} hour fcst"


def byte_ranges(entries: list[IdxEntry]) -> list[tuple[int, int | None]]:
    """(first, last) bytes to fetch; neighbouring entries are joined into one range."""
    ranges = []
    for entry in entries:
        if ranges and ranges[-1][1] == entry.start - 1:
            ranges[-1] = (ranges[-1][0], entry.end)
        else:
            ranges.append((entry.start, entry.end))
    return ranges


def exists(url: str) -> bool:
    """HEAD request; any HTTP or network error reads as absent."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30):
            return True
    except OSError:
        return False


def _fetch(url: str, range_header: str | None) -> bytes:
    headers = {"Range": range_header} if range_header else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        return response.read()


def download(
    run: datetime,
    steps: list[int],
    directory: Path,
    fetch: Callable[[str, str | None], bytes] = _fetch,
    fields=FIELDS | SKY_FIELDS,
) -> list[Path]:
    """`fields` of each step into f{step:03}.grib2. A failure or a step without one of `FIELDS` stops the run;
    the next hour tries again. A step without a sky field goes on without it."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for step in steps:
        url = file_url(run, step)
        entries = [entry for entry in parse_idx(fetch(url + ".idx", None).decode()) if wanted(entry, step, fields)]
        missing = fields - {(entry.name, entry.level) for entry in entries}
        if missing & FIELDS:
            raise ValueError(f"step {step}: no {sorted(missing & FIELDS)} in the .idx")
        if missing:
            print(f"step {step}: no {sorted(missing)} in the .idx")
        path = directory / f"f{step:03}.grib2"
        with open(path, "wb") as file:
            for start, end in byte_ranges(entries):
                file.write(fetch(url, f"bytes={start}-{'' if end is None else end}"))
        paths.append(path)
    return paths


def read_fields(paths: list[Path]) -> dict[tuple[str, int, int], np.ndarray]:
    """(name, level, step) → normalized (721, 1440) globe; level 0 for surface fields."""
    fields = {}
    for path in paths:
        with open(path, "rb") as file:
            while (handle := eccodes.codes_grib_new_from_file(file)) is not None:
                key = _key(handle)
                try:
                    fields[key] = _read_message(handle, key)
                except ValueError as error:
                    if _wind_needs(*key[:2]):
                        raise
                    print(f"skip {error}")
                finally:
                    eccodes.codes_release(handle)
    return fields


def _wind_needs(name: str, level: int) -> bool:
    """A field the wind tiles read: a broken one stops the build; any other is skipped."""
    if level == 0:
        return name in SFC_WIND
    return name in PL_WIND and str(level) in LEVELS


def _key(handle) -> tuple[str, int, int]:
    get = lambda name: eccodes.codes_get(handle, name)
    short_name, kind = get("shortName"), get("typeOfLevel")
    level = get("level") if kind == "isobaricInhPa" else 0
    return NAMES.get((short_name, kind), short_name), level, eccodes.codes_get_long(handle, "endStep")


def _read_message(handle, key: tuple[str, int, int]) -> np.ndarray:
    get = lambda name: eccodes.codes_get(handle, name)
    first_lat = get("latitudeOfFirstGridPointInDegrees")
    if (
        get("Ni") != NI
        or get("Nj") != NJ
        or get("iDirectionIncrementInDegrees") != STEP_DEG
        or get("jDirectionIncrementInDegrees") != STEP_DEG
        or abs(first_lat) != 90
    ):
        raise ValueError(f"{' '.join(map(str, key))}: not a global {STEP_DEG}° {NI} × {NJ} grid from a pole")
    if get("iScansNegatively") != 0 or get("jScansPositively") != (1 if first_lat < 0 else 0):
        raise ValueError(f"{' '.join(map(str, key))}: scanned against the first point")
    values = eccodes.codes_get_values(handle).reshape(NJ, NI)
    known = values
    if get("bitmapPresent"):
        if key[0] in PL_WIND + SFC_WIND:
            raise ValueError(f"{' '.join(map(str, key))}: has missing values")
        gaps = values == get("missingValue")
        known = values[~gaps]
        values[gaps] = np.nan
    if not np.isfinite(known).all():
        raise ValueError(f"{' '.join(map(str, key))}: non-finite values")
    low, high = RANGES.get(key[0], (-np.inf, np.inf))
    if (key[0] in WINDS and np.abs(known).max() >= WIND_LIMIT) or known.min() < low or known.max() > high:
        raise ValueError(f"{' '.join(map(str, key))}: values out of range")
    return normalize(values, first_lat, get("longitudeOfFirstGridPointInDegrees"))
