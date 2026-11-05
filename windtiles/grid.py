"""Global 0.25° grid and its slicing into 181 × 181 node tiles."""

import numpy as np

STEP_DEG = 0.25
NI = 1440  # longitudes, 180W … 179.75E
NJ = 721  # latitudes, 90S … 90N
CELLS = 180  # cells per tile side
NODES = 181  # nodes per tile side

ZOOMS: dict[int, int] = {1: 4, 2: 2, 3: 1}  # zoom → stride in 0.25° nodes
LEVELS: list[str] = ["sfc", "850", "700", "500", "400", "300", "250", "200", "150"]


def tile_grid(zoom: int) -> tuple[int, int]:
    """(rows, columns) of tiles covering the globe at this zoom."""
    span = CELLS * ZOOMS[zoom]
    return (NJ - 1) // span, NI // span


def normalize(values: np.ndarray, first_lat: float, first_lon: float) -> np.ndarray:
    """GRIB-ordered (721, 1440) field → float32 with row 0 at 90S and column 0 at 180W."""
    if first_lat > 0:
        values = values[::-1]
    shift = round(((first_lon + 180) % 360) / STEP_DEG)
    return np.roll(values, shift, axis=1).astype(np.float32)


def tile_nodes(globe: np.ndarray, zoom: int, row: int, column: int) -> np.ndarray:
    """(181, 181) nodes of one tile from a normalized globe; row 0 is the southern edge."""
    k = ZOOMS[zoom]
    south = row * CELLS * k
    columns = (column * CELLS * k + np.arange(NODES) * k) % NI
    return globe[south : south + CELLS * k + 1 : k][:, columns]
