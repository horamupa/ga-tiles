import numpy as np
from windtiles.grid import NI, NJ, normalize, tile_grid, tile_nodes

def lon_lat_globe():
    """globe[j, i] = (lat, lon) of node j from 90S, i from 180W, packed as lat * 1000 + lon."""
    lat = -90 + 0.25 * np.arange(NJ)[:, None]
    lon = -180 + 0.25 * np.arange(NI)[None, :]
    return (lat * 1000 + lon).astype(np.float64)

def test_tile_counts():
    assert [r * c for r, c in map(tile_grid, (1, 2, 3))] == [2, 8, 32]

def test_normalize_from_0_to_360():
    # GRIB order: north first, longitudes 0…359.75
    lat = 90 - 0.25 * np.arange(NJ)[:, None]
    lon = (0.25 * np.arange(NI)[None, :] + 180) % 360 - 180
    out = normalize(lat * 1000 + lon, first_lat=90, first_lon=0)
    assert np.array_equal(out, lon_lat_globe().astype(np.float32))

def test_normalize_from_minus_180():
    lat = 90 - 0.25 * np.arange(NJ)[:, None]
    lon = -180 + 0.25 * np.arange(NI)[None, :]
    assert np.array_equal(normalize(lat * 1000 + lon, 90, -180), lon_lat_globe().astype(np.float32))

def test_tile_nodes_corners():
    nodes = tile_nodes(lon_lat_globe(), zoom=3, row=2, column=4)  # 0…45N, 0…45E
    assert nodes.shape == (181, 181)
    assert nodes[0, 0] == 0 and nodes[180, 180] == 45 * 1000 + 45

def test_east_edge_wraps():
    nodes = tile_nodes(lon_lat_globe(), zoom=3, row=2, column=7)  # 135…180E
    assert nodes[0, 180] == -180  # 180E is 180W

def test_top_row_reaches_the_pole():
    nodes = tile_nodes(lon_lat_globe(), zoom=1, row=0, column=1)  # 1°, 90S…90N, 0…180E
    assert nodes[180, 0] == 90 * 1000 and nodes[0, 1] == -90 * 1000 + 1

def test_coarse_zoom_takes_every_other_node():
    nodes = tile_nodes(lon_lat_globe(), zoom=2, row=1, column=2)  # 0.5°, 0…90N, 0…90E
    assert nodes[1, 1] == 0.5 * 1000 + 0.5
