import os
from datetime import datetime, timezone

import numpy as np
import pytest
from omfiles import OmFileWriter
from pyproj import CRS, Transformer

from windtiles import om_source
from windtiles.om_source import grid_map, latest_run, read, regrid

SPHERE = 'BASEGEOGCRS["GCS_Sphere", DATUM["D_Sphere", ELLIPSOID["Sphere",6371229.0,0.0]]]'


def latlon_wkt(south, west, north, east):
    return (
        'GEOGCRS["WGS 84", DATUM["World Geodetic System 1984", ELLIPSOID["WGS 84",6378137,298.257223563]],'
        ' CS[ellipsoidal,2], AXIS["latitude",north], AXIS["longitude",east],'
        ' ANGLEUNIT["degree",0.0174532925199433]'
        f' USAGE[SCOPE["grid"], BBOX[{south},{west},{north},{east}]]]'
    )


def lambert_wkt(corners):
    """HRRR's projection with the given SW / NE grid points (south, west, north, east)."""
    return (
        f'PROJCRS["Lambert Conic Conformal", {SPHERE},'
        ' CONVERSION["Lambert Conic Conformal", METHOD["Lambert Conic Conformal (2SP)"],'
        ' PARAMETER["Latitude of 1st standard parallel",38.5], PARAMETER["Latitude of 2nd standard parallel",38.5],'
        ' PARAMETER["Latitude of false origin",0.0], PARAMETER["Longitude of false origin",-97.5]],'
        ' CS[Cartesian,2], AXIS["easting",east], AXIS["northing",north], LENGTHUNIT["metre",1],'
        f' USAGE[SCOPE["grid"], BBOX[{",".join(map(str, corners))}]]]'
    )


def gaussian_wkt(n):
    return (
        'GEOGCRS["Reduced Gaussian Grid", DATUM["World Geodetic System 1984", ELLIPSOID["WGS 84",6378137,298.257223563]],'
        ' CS[ellipsoidal,2], AXIS["latitude",north], AXIS["longitude",east], ANGLEUNIT["degree",0.0174532925199433],'
        f' REMARK["Reduced Gaussian Grid O{n} (ECMWF)"], USAGE[SCOPE["grid"], BBOX[-90,-180.0,90,180]]]'
    )


def nodes(grid, zoom):
    step = 2.0 ** (1 - zoom)
    rows, columns = grid.shape
    return np.meshgrid(grid.south + step * np.arange(rows), grid.west + step * np.arange(columns), indexing="ij")


def test_window_starts_on_tile_boundary_and_covers_domain():
    zoom = 7
    side = 180 * 2.0 ** (1 - zoom)
    grid = grid_map(latlon_wkt(43.18, -3.94, 58.08, 20.34), (746, 1215), zoom)  # ICON-D2
    rows, columns = grid.shape
    assert ((grid.south + 90) / side).is_integer() and ((grid.west + 180) / side).is_integer()
    assert (rows - 1) % 180 == 0 and (columns - 1) % 180 == 0
    assert grid.south <= 43.18 < grid.south + side
    assert grid.west <= -3.94 < grid.west + side
    assert grid.south + (rows - 1) * side / 180 >= 58.08
    assert grid.west + (columns - 1) * side / 180 >= 20.34


def test_regular_grid_reproduces_linear_field_and_is_nan_outside():
    lat = 40 + 0.5 * np.arange(21)
    lon = 0.5 * np.arange(31)
    data = (2 * lat[:, None] + 3 * lon[None, :]).astype(np.float32)  # row 0 south, column 0 west
    zoom = 5
    grid = grid_map(latlon_wkt(40, 0, 50, 15), data.shape, zoom)
    out = regrid(data, grid)
    la, lo = nodes(grid, zoom)
    inside = (la >= 40) & (la <= 50) & (lo >= 0) & (lo <= 15)
    assert out.dtype == np.float32 and out.shape == grid.shape
    np.testing.assert_allclose(out[inside], (2 * la + 3 * lo)[inside], atol=1e-3)
    assert np.isnan(out[~inside]).all()


def test_global_regular_grid_wraps_across_the_gap_to_column_zero():
    # ICON-like: 179 columns every 2° from 180°W to 176°E, the last cell spans 176°E → 180°.
    nx, ny = 179, 91
    data = np.tile(np.linspace(-90, 90, ny, dtype=np.float32)[:, None], (1, nx))
    zoom = 1
    grid = grid_map(latlon_wkt(-90, -180, 90, 176), (ny, nx), zoom)
    la, lo = nodes(grid, zoom)
    assert grid.shape == (181, 361) and (grid.south, grid.west) == (-90, -180)
    np.testing.assert_allclose(regrid(data, grid), la, atol=1e-4)  # no NaN anywhere, longitude wraps

    ramp = np.zeros((ny, nx), np.float32)
    ramp[:, -1] = 1  # 176°E is 1, 180° (column 0) is 0
    out = regrid(ramp, grid)
    np.testing.assert_allclose(out[:, lo[0] == 178], 0.5, atol=1e-6)
    np.testing.assert_allclose(out[:, lo[0] == 176], 1, atol=1e-6)
    np.testing.assert_allclose(out[:, lo[0] == 180], 0, atol=1e-6)


def test_projected_grid_reproduces_projected_coordinate():
    nx, ny, dx = 80, 60, 30_000.0
    x0, y0 = -1_200_000.0, 3_200_000.0
    crs = CRS.from_wkt(lambert_wkt((0, 0, 1, 1)))
    to_latlon = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    west, south = to_latlon.transform(x0, y0)
    east, north = to_latlon.transform(x0 + (nx - 1) * dx, y0 + (ny - 1) * dx)
    wkt = lambert_wkt((south, west, north, east))
    data = np.tile((x0 + dx * np.arange(nx, dtype=np.float64))[None, :], (ny, 1)).astype(np.float32)
    zoom = 3
    grid = grid_map(wkt, (ny, nx), zoom)
    out = regrid(data, grid)
    la, lo = nodes(grid, zoom)
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lo, la)
    inside = (x >= x0 + 1) & (x <= x0 + (nx - 1) * dx - 1) & (y >= y0 + 1) & (y <= y0 + (ny - 1) * dx - 1)
    outside = (x < x0 - 1) | (x > x0 + (nx - 1) * dx + 1) | (y < y0 - 1) | (y > y0 + (ny - 1) * dx + 1)
    assert inside.sum() > 1000 and outside.sum() > 1000
    np.testing.assert_allclose(out[inside], x[inside], rtol=0, atol=1)  # float32 holds ~0.1 m here
    assert np.isnan(out[outside]).all()


def gaussian_points(n):
    """Latitude and longitude of every point of an O-grid, north line first, from 0°E."""
    dy = 180 / (2 * n + 0.5)
    lats, lons = [], []
    for y in range(2 * n):
        count = 20 + 4 * min(y, 2 * n - 1 - y)
        lats += [(n - y - 1) * dy + dy / 2] * count
        lons += list(np.arange(count) * 360 / count)
    return np.array(lats), np.array(lons)


def test_gaussian_grid_interpolates_along_and_between_lines():
    n = 32
    lat, lon = gaussian_points(n)
    data = (lat + 10 * np.cos(np.radians(lon))).astype(np.float32)
    zoom = 1
    grid = grid_map(gaussian_wkt(n), (1, lat.size), zoom)
    assert (grid.south, grid.west, grid.shape) == (-90, -180, (181, 361))
    out = regrid(data, grid)
    la, lo = nodes(grid, zoom)
    assert not np.isnan(out).any()
    middle = np.abs(la) <= 60  # lines hold ≥ 80 points there: cos error ≤ 0.03
    np.testing.assert_allclose(out[middle], (la + 10 * np.cos(np.radians(lo)))[middle], atol=0.05)
    pole_line = (n - 1) * 180 / (2 * n + 0.5) + 90 / (2 * n + 0.5)
    assert np.nanmax(out) <= pole_line + 10 + 1e-3  # beyond the outermost line: no extrapolation


def test_gaussian_grid_rejects_wrong_point_count():
    with pytest.raises(ValueError):
        grid_map(gaussian_wkt(32), (1, 1000), 1)


def test_read_downloads_only_the_requested_variable(tmp_path, monkeypatch):
    path = tmp_path / "frame.om"
    wkt = latlon_wkt(40, 0, 50, 15)
    rng = np.random.default_rng(0)
    writer = OmFileWriter(str(path))
    children = [writer.write_scalar(wkt, name="crs_wkt")]
    for i in range(20):
        noise = rng.uniform(0, 100, (201, 301)).astype(np.float32)  # incompressible, ~120 KB each
        unit = writer.write_scalar("%", name="unit")
        children.append(writer.write_array(noise, chunks=[32, 32], scale_factor=100, name=f"noise_{i}", children=[unit]))
    field = (np.arange(201)[:, None] + np.zeros(301)[None, :]).astype(np.float32)
    unit = writer.write_scalar("m", name="unit")
    children.append(writer.write_array(field, chunks=[32, 32], scale_factor=10, name="height", children=[unit]))
    writer.close(writer.write_group("", children))
    blob = path.read_bytes()

    def fetch(url, byte_range):
        start, stop = byte_range.removeprefix("bytes=").split("-")
        part = blob[-int(stop):] if start == "" else blob[int(start) : int(stop) + 1]
        return part, len(blob)

    monkeypatch.setattr(om_source, "_fetch", fetch)
    run = datetime(2026, 11, 10, 12, tzinfo=timezone.utc)
    fields = read("model", run, run, ["height"], zoom=5)
    assert fields.units == {"height": "m"}
    assert fields.downloaded < len(blob) / 3
    out = fields.values["height"]
    step = 2.0 ** (1 - 5)
    row = round((45 - fields.south) / step)
    column = round((7.5 - fields.west) / step)
    assert out[row, column] == pytest.approx(100, abs=0.1)  # 45°N is source row 100

    doubled = read("model", run, run, ["height"], zoom=5, prepare=lambda source: source.update(height=source["height"] * 2))
    assert doubled.values["height"][row, column] == pytest.approx(200, abs=0.2)  # changed on the source grid


@pytest.mark.skipif(not os.environ.get("OM_NETWORK"), reason="network: set OM_NETWORK=1")
def test_latest_icon_d2_frame_from_bucket():
    run = latest_run("dwd_icon_d2", hours=(0, 6, 12, 18))
    assert run.time.hour in (0, 6, 12, 18) and len(run.valid_times) == 49
    fields = read("dwd_icon_d2", run.time, run.valid_times[6], ["cloud_cover"], zoom=7)
    cover = fields.values["cloud_cover"]
    assert fields.units == {"cloud_cover": "%"}
    assert 0.2 < np.isnan(cover).mean() < 0.6 and 0 <= np.nanmin(cover) and np.nanmax(cover) <= 100.01


def test_infinity_is_kept_out_of_the_blend():
    # UKV cloud_base with no cloud as +inf. Columns every 1° from 0°E: none, 300 m, 300 m, 900 m, none, none.
    inf = np.inf
    data = np.tile(np.array([inf, 300, 300, 900, inf, inf], np.float32), (5, 1))
    grid = grid_map(latlon_wkt(40, 0, 44, 5), data.shape, zoom=2)  # nodes every 0.5°
    out = regrid(data, grid)
    la, lo = nodes(grid, 2)
    row = out[la[:, 0] == 42][0]
    at = lambda lon: row[lo[0] == lon][0]
    assert at(0) == inf and at(0.5) == 300 and at(1.5) == 300 and at(2.5) == 600  # no blend of 300 m with "none"
    assert at(3.5) == 900 and at(4.5) == inf and np.isnan(at(5.5))  # between two clear cells: none; outside: NaN
    assert regrid(np.where(data == inf, 0, data), grid)[la == 42][1] == 150  # a 0 for "none" would blend


def test_rotated_grid_reproduces_rotated_longitude():
    # HRDPS's rotated pole; a 0.5° grid in rotated coordinates over the Canadian Prairies.
    wkt_head = (
        'GEOGCRS["Rotated Lat/Lon", BASEGEOGCRS["GCS_Sphere", DATUM["D_Sphere", ELLIPSOID["Sphere",6371229.0,0.0]]],'
        ' DERIVINGCONVERSION["Rotated Lat/Lon", METHOD["PROJ ob_tran o_proj=longlat"], PARAMETER["o_lon_p",0],'
        ' PARAMETER["o_lat_p",36.0885], PARAMETER["lon_0",245.305]], CS[ellipsoidal,2], AXIS["latitude",north],'
        ' AXIS["longitude",east], ANGLEUNIT["degree",0.0174532925199433]'
    )
    crs = CRS.from_wkt(wkt_head + ', USAGE[SCOPE["grid"], BBOX[0,0,1,1]]]')
    nx, ny, x0, y0, d = 41, 31, 0.0, 0.0, 0.5
    to_latlon = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    west, south = to_latlon.transform(x0, y0)
    east, north = to_latlon.transform(x0 + (nx - 1) * d, y0 + (ny - 1) * d)
    wkt = wkt_head + f', USAGE[SCOPE["grid"], BBOX[{south},{west},{north},{east}]]]'
    data = np.tile((x0 + d * np.arange(nx))[None, :], (ny, 1)).astype(np.float32)
    grid = grid_map(wkt, (ny, nx), 4)
    out = regrid(data, grid)
    la, lo = nodes(grid, 4)
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lo, la)
    inside = (x > x0 + 0.01) & (x < x0 + (nx - 1) * d - 0.01) & (y > y0 + 0.01) & (y < y0 + (ny - 1) * d - 0.01)
    assert inside.sum() > 1000 and np.isnan(out[~inside & ((x < x0 - 0.01) | (y < y0 - 0.01))]).all()
    np.testing.assert_allclose(out[inside], x[inside], atol=1e-3)
