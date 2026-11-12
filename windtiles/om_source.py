"""Open-Meteo open-data bucket (.om files) → float32 fields on the tile grid.

One `.om` file per valid time holds every variable of a model. Only the bytes of
the requested variables are downloaded, through HTTP range requests.

Tile grid at zoom z: nodes every `2^(1−z)°`, tiles of 180 × 180 cells counted
from 90°S / 180°W, row 0 southernmost, column 0 westernmost.

Regridding is bilinear on all source grid kinds (regular latitude/longitude,
projected, rotated latitude/longitude, ECMWF reduced Gaussian). It is meant for each model's own maximum zoom,
where the source is as fine as the target or up to ~1.7× finer; coarser zooms are
aggregated from that array by the tile builder.
"""

import json
import math
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from functools import lru_cache
from typing import NamedTuple

import numpy as np
from omfiles import OmFileReader
from pyproj import CRS, Transformer

BUCKET = "https://openmeteo.s3.amazonaws.com/data_spatial"
TAIL = 262_144  # metadata and lookup tables sit at the end of a file
READ_AHEAD = 131_072  # first range request of a variable; doubles while reads stay sequential
_GRID_LOCK = threading.Lock()


class Run(NamedTuple):
    time: datetime
    valid_times: list[datetime]
    variables: list[str]
    modified: datetime  # last_modified_time of the run's metadata


class Fields(NamedTuple):
    south: float  # latitude of row 0, on a tile boundary
    west: float  # longitude of column 0, on a tile boundary
    values: dict[str, np.ndarray]  # float32 (rows, columns), NaN outside the model domain
    units: dict[str, str]  # as stored in the file, e.g. "%", "m", "J/kg"
    downloaded: int  # bytes fetched for this call


class GridMap(NamedTuple):
    south: float
    west: float
    shape: tuple[int, int]
    index: np.ndarray  # (4, rows·columns) int32 flat indices into the source array
    weight: np.ndarray  # (4, rows·columns) float32, NaN outside the model domain


def run_url(model: str, run: datetime) -> str:
    return f"{BUCKET}/{model}/{run:%Y/%m/%d/%H%M}Z/"


def file_url(model: str, run: datetime, valid: datetime) -> str:
    return f"{run_url(model, run)}{valid:%Y-%m-%dT%H%M}.om"


def latest_run(model: str, hours: tuple[int, ...] | None = None) -> Run:
    """Latest complete run; with `hours`, the latest complete run starting at one of those UTC hours."""
    meta = _json(f"{BUCKET}/{model}/latest.json")
    run = _time(meta["reference_time"])
    for _ in range(25):
        if meta and meta["completed"] and (hours is None or run.hour in hours):
            return Run(run, [_time(t) for t in meta["valid_times"]], meta["variables"], _time(meta["last_modified_time"]))
        run -= timedelta(hours=1)
        meta = _json(run_url(model, run) + "meta.json")
    raise LookupError(f"{model}: no complete run at hours {hours} in the last day")


def read(
    model: str, run: datetime, valid: datetime, variables: list[str], zoom: int, prepare=None
) -> Fields:
    """Variables of one valid time on the tile grid of `zoom`, in the bucket's own units. `prepare`, if given, may
    replace arrays of the name → array dict on the model's own grid before regridding, e.g. with +inf for "none" (see
    `regrid`). Safe to call from several threads."""
    url = file_url(model, run, valid)
    file = _RangeFile(url)
    root = OmFileReader.from_fsspec(file, url)
    crs_wkt = root.get_child_by_name("crs_wkt").read_scalar()
    source, values, units = {}, {}, {}
    for name in variables:
        variable = root.get_child_by_name(name)
        source[name] = variable[...]
        units[name] = variable.get_child_by_name("unit").read_scalar()
    if prepare:
        prepare(source)
    for name, data in source.items():
        with _GRID_LOCK:  # one index map per grid, not one per thread: ECMWF's at zoom 5 takes gigabytes to build
            grid = grid_map(crs_wkt, data.shape, zoom)
        values[name] = regrid(data, grid)
    return Fields(grid.south, grid.west, values, units, file.downloaded)


def regrid(data: np.ndarray, grid: GridMap) -> np.ndarray:
    """Bilinear. A source +inf means "none" (UKV: no cloud base) and is not blended: a node takes the weighted mean of
    its finite corners, and +inf only where all four corners are +inf."""
    corners = data.reshape(-1)[grid.index]
    if not np.isposinf(data).any():
        return (grid.weight * corners).sum(axis=0).reshape(grid.shape).astype(np.float32)
    none = np.isposinf(corners)
    weight = grid.weight * ~none  # NaN outside the domain stays NaN
    total = weight.sum(axis=0)
    with np.errstate(invalid="ignore"):
        out = np.where(total == 0, np.inf, (weight * np.where(none, 0, corners)).sum(axis=0) / total)
    return out.reshape(grid.shape).astype(np.float32)


@lru_cache(maxsize=4)
def grid_map(crs_wkt: str, shape: tuple[int, int], zoom: int) -> GridMap:
    """Bilinear index map from a source grid to the tile-aligned window around its domain."""
    step = 2.0 ** (1 - zoom)
    side = 180 * step
    gaussian = "Gaussian" in crs_wkt
    south, north, west, east = (-90, 90, -180, 180) if gaussian else _extent(crs_wkt, shape)
    row0 = max(0, math.floor((south + 90) / side))
    row1 = min(round(1 / step), math.ceil((north + 90) / side))
    col0 = max(0, math.floor((west + 180) / side))
    col1 = min(round(2 / step), math.ceil((east + 180) / side))
    south, west = row0 * side - 90, col0 * side - 180
    rows, columns = (row1 - row0) * 180 + 1, (col1 - col0) * 180 + 1
    lat, lon = np.meshgrid(south + step * np.arange(rows), west + step * np.arange(columns), indexing="ij")
    lat, lon = lat.reshape(-1), lon.reshape(-1)
    if gaussian:
        index, weight = _gaussian(crs_wkt, shape[-1], lat, lon)
    else:
        index, weight = _bilinear(*_fractional(crs_wkt, shape, lat, lon), shape)
    return GridMap(south, west, (rows, columns), index.astype(np.int32), weight.astype(np.float32))


def _extent(crs_wkt: str, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """South, north, west, east of a regular or projected grid."""
    crs = CRS.from_wkt(crs_wkt)
    area = crs.area_of_use
    if crs.is_geographic and not crs.is_derived:  # a rotated grid (HRDPS) goes the projected way
        cyclic = _wraps(area, shape[1])
        return area.south, area.north, (-180 if cyclic else area.west), (180 if cyclic else area.east)
    ny, nx = shape
    fy = np.concatenate([np.zeros(nx), np.full(nx, ny - 1), np.arange(ny), np.arange(ny)])
    fx = np.concatenate([np.arange(nx), np.arange(nx), np.zeros(ny), np.full(ny, nx - 1)])
    x0, y0, x1, y1 = _corners(crs)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(
        x0 + fx * (x1 - x0) / (nx - 1), y0 + fy * (y1 - y0) / (ny - 1)
    )
    return lat.min(), lat.max(), lon.min(), lon.max()


def _wraps(area, nx: int) -> bool:
    """A latitude/longitude grid whose columns close the circle, with a gap of up to two cells."""
    dx = (area.east - area.west) / (nx - 1)
    return area.east - area.west + 2 * dx >= 360 - dx / 2


def _corners(crs: CRS) -> tuple[float, float, float, float]:
    """Projected x, y of the first (south-west) and last (north-east) grid points.

    The bucket's BBOX holds the latitude/longitude of those two points, not a bounding box."""
    area = crs.area_of_use
    to_grid = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x0, y0 = to_grid.transform(area.west, area.south)
    x1, y1 = to_grid.transform(area.east, area.north)
    return x0, y0, x1, y1


def _fractional(crs_wkt: str, shape: tuple[int, int], lat: np.ndarray, lon: np.ndarray):
    """Fractional source (row, column) of each target node, and whether columns wrap around the globe."""
    crs = CRS.from_wkt(crs_wkt)
    ny, nx = shape
    if crs.is_geographic and not crs.is_derived:
        area = crs.area_of_use
        dx = (area.east - area.west) / (nx - 1)
        fy = (lat - area.south) / ((area.north - area.south) / (ny - 1))
        if not _wraps(area, nx):
            return fy, (lon - area.west) / dx, False
        # ICON's global grid stops at 179.75°E with 2879 columns: the last cell spans the gap back to column 0.
        offset = (lon - area.west) % 360
        last = (nx - 1) * dx
        fx = np.where(offset <= last, offset / dx, nx - 1 + (offset - last) / (360 - last))
        return fy, fx, True
    x0, y0, x1, y1 = _corners(crs)
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    return (y - y0) / (y1 - y0) * (ny - 1), (x - x0) / (x1 - x0) * (nx - 1), False


def _bilinear(fy: np.ndarray, fx: np.ndarray, cyclic: bool, shape: tuple[int, int]):
    ny, nx = shape
    eps = 1e-6
    x_max = nx if cyclic else nx - 1  # with wrap-around, column nx is column 0
    inside = (fy >= -eps) & (fy <= ny - 1 + eps) & (fx >= -eps) & (fx <= x_max + eps)
    fy, fx = np.clip(fy, 0, ny - 1), np.clip(fx, 0, x_max)
    y0 = np.minimum(np.floor(fy), ny - 2).astype(np.int64)
    x0 = np.minimum(np.floor(fx), x_max - 1).astype(np.int64)
    wy, wx = fy - y0, fx - x0
    x1 = (x0 + 1) % nx
    index = np.stack([y0 * nx + x0, y0 * nx + x1, (y0 + 1) * nx + x0, (y0 + 1) * nx + x1])
    weight = np.stack([(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx])
    weight[:, ~inside] = np.nan
    return index, weight


def _gaussian(crs_wkt: str, count: int, lat: np.ndarray, lon: np.ndarray):
    """Bilinear on an octahedral reduced Gaussian grid (ECMWF O1280).

    Lines run north to south; line y holds 20 + 4·min(y, 2N−1−y) points from 0°E eastward.
    Latitudes use Open-Meteo's own spacing 180 / (2N + 0.5), within 0.01° of the true Gaussian ones."""
    n = int(re.search(r"\bO(\d+)\b", crs_wkt).group(1))
    if count != 4 * n * (n + 9):
        raise ValueError(f"O{n} grid has {4 * n * (n + 9)} points, the file has {count}")
    lines = np.arange(2 * n)
    per_line = 20 + 4 * np.minimum(lines, 2 * n - 1 - lines)
    first = np.concatenate([[0], np.cumsum(per_line)[:-1]])
    dy = 180 / (2 * n + 0.5)
    fy = n - 1 - (lat - dy / 2) / dy
    y0 = np.clip(np.floor(fy), 0, 2 * n - 2).astype(np.int64)
    wy = np.clip(fy - y0, 0, 1)
    index, weight = [], []
    for y, w in ((y0, 1 - wy), (y0 + 1, wy)):
        fx = (lon % 360) / 360 * per_line[y]
        x0 = np.floor(fx).astype(np.int64)
        wx = fx - x0
        index += [first[y] + x0 % per_line[y], first[y] + (x0 + 1) % per_line[y]]
        weight += [w * (1 - wx), w * wx]
    return np.stack(index), np.stack(weight)


class _RangeFile:
    """The `size` / `cat_file` pair `OmFileReader.from_fsspec` reads through, over HTTP ranges.

    The reader asks for one compressed 32 × 32 chunk (~1 KB) at a time, in ascending
    order within a variable, and each request costs a round trip. So a miss reads
    ahead, doubling the read-ahead while reads stay sequential: a variable of n bytes
    costs about log2(n / 128 KB) requests and at most 2n bytes."""

    def __init__(self, url: str):
        self.url = url
        tail, self.total = _fetch(url, f"bytes=-{TAIL}")
        self.blocks = [(self.total - len(tail), tail)]
        self.downloaded = len(tail)
        self.ahead, self.last = READ_AHEAD, (0, 0)

    def size(self, path: str) -> int:
        return self.total

    def cat_file(self, path: str, start: int, end: int, **kwargs) -> bytes:
        for offset, data in self.blocks:
            if offset <= start and end <= offset + len(data):
                return data[start - offset : end - offset]
        self.ahead = self.ahead * 2 if self.last[0] <= start <= self.last[1] else READ_AHEAD
        stop = min(self.total, max(end, start + self.ahead))
        data, _ = _fetch(self.url, f"bytes={start}-{stop - 1}")
        self.blocks.append((start, data))
        self.downloaded += len(data)
        self.last = (start, stop)
        return data[: end - start]


def _fetch(url: str, byte_range: str) -> tuple[bytes, int]:
    """Body of a range request and the file's total size."""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": byte_range}), timeout=60) as r:
                return r.read(), int(r.headers["Content-Range"].rsplit("/", 1)[1])
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def _json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as error:
        if error.code in (403, 404):  # S3 answers 403 for a missing key without list rights
            return None
        raise


def _time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))
