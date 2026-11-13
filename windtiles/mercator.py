"""Coloured Web Mercator XYZ tiles of a precipitation sum on a regular latitude-longitude grid."""

import io
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 256  # pixels per tile side
MAX_LATITUDE = 85.0511  # Web Mercator edge, degrees

# (threshold mm, RGBA): a value ≥ threshold k and < threshold k + 1 takes colour k, ≥ 100 the last; below 0.5 nothing.
LEGEND: list[tuple[float, tuple[int, int, int, int]]] = [
    (0.5, (160, 210, 255, 191)),
    (1, (100, 170, 245, 191)),
    (2, (40, 120, 230, 191)),
    (5, (20, 70, 200, 191)),
    (10, (40, 170, 70, 191)),
    (15, (130, 200, 40, 191)),
    (20, (240, 220, 30, 191)),
    (30, (250, 150, 20, 191)),
    (50, (230, 50, 30, 191)),
    (75, (190, 20, 100, 191)),
    (100, (220, 50, 220, 191)),
]
NO_DATA: tuple[int, int, int, int] = (128, 128, 128, 77)  # a NaN node inside the rectangle of rows and columns that hold data
# Palette: index 0 transparent, k + 1 = LEGEND[k], then NO_DATA.
_COLOURS = [(0, 0, 0, 0)] + [rgba for _, rgba in LEGEND] + [NO_DATA]
_RGB = bytes(c for rgba in _COLOURS for c in rgba[:3])
_ALPHA = bytes(rgba[3] for rgba in _COLOURS)


def write_mercator_tiles(
    directory: Path, field: np.ndarray, south: float, west: float, step: float, max_zoom: int
) -> int:
    """Tiles of zooms 0…`max_zoom` into `directory`/{z}/{x}/{y}.png (x from 180°W, y from the north); returns their
    count.

    `field` is a (rows, columns) array in mm, NaN for no data, on nodes `step`° apart: row 0 at latitude `south`,
    column 0 at longitude `west`. A field 360° wide wraps in longitude; any other is a region with nothing outside it.
    A pixel takes the node nearest to its centre and the `LEGEND` colour of its value; below 0.5 mm or outside the
    field it is transparent. A NaN node is `NO_DATA` inside the rectangle of rows and columns that hold data, and
    transparent outside it (the padding around a regional model's grid). A tile with no coloured pixel is not
    written."""
    rows, columns = field.shape
    periodic = round(columns * step, 6) == 360
    thresholds = np.array([mm for mm, _ in LEGEND])
    # Palette index per node; one extra row and column of 0 (transparent) that index −1 (outside) lands on.
    classes = np.zeros((rows + 1, columns + 1), dtype=np.uint8)
    classes[:rows, :columns] = np.searchsorted(thresholds, np.nan_to_num(field, nan=0.0), side="right")
    missing = np.isnan(field)
    data_rows, data_columns = np.flatnonzero(~missing.all(axis=1)), np.flatnonzero(~missing.all(axis=0))
    if len(data_rows):
        box = slice(data_rows[0], data_rows[-1] + 1), slice(data_columns[0], data_columns[-1] + 1)
        classes[box][missing[box]] = len(_COLOURS) - 1
    count = 0
    for z in range(max_zoom + 1):
        pixels = 2**z * SIZE  # pixels along each side of the world at zoom z
        centres = (np.arange(pixels) + 0.5) / pixels
        lon = centres * 360 - 180
        lat = np.clip(np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * centres)))), -MAX_LATITUDE, MAX_LATITUDE)
        row_of = _nearest(lat - south, step, rows)
        offset = (lon - west + step / 2) % 360 - step / 2  # degrees east of `west`, a region crossing 180° included
        column_of = _nearest(offset, step, columns, periodic)
        tiles_y = [y for y in range(2**z) if (row_of[y * SIZE : (y + 1) * SIZE] >= 0).any()]
        tiles_x = [x for x in range(2**z) if (column_of[x * SIZE : (x + 1) * SIZE] >= 0).any()]
        for y in tiles_y:
            tile_rows = row_of[y * SIZE : (y + 1) * SIZE]
            for x in tiles_x:
                tile = classes[tile_rows[:, None], column_of[None, x * SIZE : (x + 1) * SIZE]]
                if not tile.any():
                    continue
                path = directory / f"{z}/{x}/{y}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(_png(tile))
                count += 1
    return count


def _nearest(offset: np.ndarray, step: float, count: int, periodic: bool = False) -> np.ndarray:
    """Index of the node nearest to each offset in degrees from node 0: modulo `count` if `periodic`, else −1 outside
    0…`count` − 1."""
    index = np.floor(offset / step + 0.5).astype(np.int64)
    if periodic:
        return index % count
    return np.where((index >= 0) & (index < count), index, -1)


def _png(classes: np.ndarray) -> bytes:
    """(256, 256) uint8 palette indices, row 0 northern → indexed-colour PNG bytes with tRNS alpha."""
    image = Image.fromarray(classes, "P")
    image.putpalette(_RGB)
    buffer = io.BytesIO()
    image.save(buffer, "PNG", transparency=_ALPHA)
    return buffer.getvalue()
