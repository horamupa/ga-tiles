"""Tiles of zooms 1…z from three channels on any regular latitude-longitude grid of 2^(1 − z)°, z = 1…7."""

from pathlib import Path

import numpy as np

from windtiles.encode import encode_tile
from windtiles.grid import CELLS, NODES

MAX_ZOOM = 7


def write_tiles(directory: Path, channels, pools, south: float, west: float, zoom: int) -> int:
    """Tiles of zooms 1…`zoom` into `directory`/{z}/{r}/{c}.png; returns their count.

    `channels` are three equal (rows, columns) arrays, NaN for no data, on nodes 2^(1 − `zoom`)° apart: row 0 at
    `south`, column 0 at `west`, both on that grid. A field 360° wide wraps in longitude; any other is a region with
    nothing outside it. A coarser zoom takes every node of its own grid, pooled over ±half its stride in fine nodes per
    channel by `pools`: "mean", "min", "max" or "node". A tile is written only if the field reaches inside it and it
    holds data; no data is byte 0 in the channel that has it alone."""
    if not 1 <= zoom <= MAX_ZOOM:
        raise ValueError(f"zoom {zoom} outside 1…{MAX_ZOOM}")
    step = 2.0 ** (1 - zoom)
    rows, columns = channels[0].shape
    periodic = columns * step == 360
    south_node, west_node = round((south + 90) / step), round((west + 180) / step)  # fine nodes from 90S and 180W
    count = 0
    for z in range(1, zoom + 1):
        k = 2 ** (zoom - z)  # fine nodes per node of zoom z
        first_row, first_column = -south_node % k, -west_node % k  # first fine node on the grid of zoom z
        coarse = [
            _pool(_pool(x, k, first_column, how, 1, periodic), k, first_row, how, 0, False)
            for x, how in zip(channels, pools)
        ]
        south_z, west_z = (south_node + first_row) // k, (west_node + first_column) // k  # nodes of zoom z
        rows_z, columns_z = coarse[0].shape
        # Tiles whose inside the field reaches: a tile that shares only its edge with the field is not written.
        tile_rows = range(max(0, south_z // CELLS), min(2 ** (z - 1), (south_z + rows_z - 2) // CELLS + 1))
        if periodic:
            tile_columns = range(2**z)
        else:
            tile_columns = range(max(0, west_z // CELLS), min(2**z, (west_z + columns_z - 2) // CELLS + 1))
        for row in tile_rows:
            i = row * CELLS - south_z + np.arange(NODES)
            for column in tile_columns:
                j = column * CELLS - west_z + np.arange(NODES)
                if periodic:
                    j %= columns_z
                inside = ((i >= 0) & (i < rows_z))[:, None] & ((j >= 0) & (j < columns_z))[None, :]
                ii, jj = np.clip(i, 0, rows_z - 1), np.clip(j, 0, columns_z - 1)
                nodes = [np.where(inside, x[ii][:, jj], np.nan) for x in coarse]
                if all(np.isnan(x).all() for x in nodes):
                    continue
                path = directory / f"{z}/{row}/{column}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(encode_tile(*nodes, per_channel=True))
                count += 1
    return count


def _pool(x: np.ndarray, k: int, first: int, how: str, axis: int, periodic: bool) -> np.ndarray:
    """Every `k`-th line of `x` along `axis` from `first`, each the `how` of the lines within ±k // 2 of it, NaN
    ignored; "node" takes the line itself."""
    x = np.moveaxis(x, axis, 0)
    centers = np.arange(first, len(x), k)
    if how == "node" or k == 1:
        return np.moveaxis(x[centers], 0, axis)
    window = centers[:, None] + np.arange(-(k // 2), k // 2 + 1)
    inside = periodic | ((window >= 0) & (window < len(x)))
    values = np.where(inside[..., None], x[window % len(x)], np.nan)
    known = ~np.isnan(values)
    with np.errstate(invalid="ignore", divide="ignore"):
        if how == "mean":
            out = np.where(known, values, 0).sum(axis=1) / known.sum(axis=1)
        elif how == "min":
            out = np.where(known, values, np.inf).min(axis=1)
        elif how == "max":
            out = np.where(known, values, -np.inf).max(axis=1)
        else:
            raise ValueError(f"pool {how!r}")
    out[~known.any(axis=1)] = np.nan
    return np.moveaxis(out.astype(x.dtype), 0, axis)
