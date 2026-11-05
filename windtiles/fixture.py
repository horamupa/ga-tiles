"""Synthetic tile the iOS tests decode: ramps in u, v, T and a no-data corner."""

import base64
import sys

import numpy as np

from windtiles.encode import encode_tile
from windtiles.grid import NODES


def fixture_fields() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """u = i − 90, v = j − 90, T = −60 + j / 9; NaN in all three for i < 10 and j < 10."""
    j, i = np.mgrid[0:NODES, 0:NODES].astype(np.float32)
    u, v, t = i - 90, j - 90, -60 + j / 9
    corner = (i < 10) & (j < 10)
    for field in (u, v, t):
        field[corner] = np.nan
    return u, v, t


def main() -> None:
    png = encode_tile(*fixture_fields())
    with open(sys.argv[1], "wb") as file:
        file.write(png)
    print(base64.b64encode(png).decode())


if __name__ == "__main__":
    main()
