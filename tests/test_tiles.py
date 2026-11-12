import io
import numpy as np
import pytest
from PIL import Image
from windtiles.encode import decode_tile
from windtiles.tiles import write_tiles

def region(value, rows=181, columns=181):
    return np.full((rows, columns), value, dtype=np.float32)

def tiles_of(directory):
    return sorted(str(path.relative_to(directory)) for path in directory.rglob("*.png"))

def read(directory, name):
    return decode_tile((directory / name).read_bytes(), per_channel=True)

def test_a_region_writes_its_own_tile_at_every_zoom(tmp_path):
    # One z = 5 tile: 45…56.25N, 0…11.25E, nodes 1/16° apart.
    count = write_tiles(tmp_path, [region(10.0), region(20.0), region(30.0)], ("mean", "min", "max"), 45, 0, zoom=5)
    assert tiles_of(tmp_path) == ["1/0/1.png", "2/1/2.png", "3/3/4.png", "4/6/8.png", "5/12/16.png"]
    assert count == 5
    r, g, b = read(tmp_path, "4/6/8.png")  # 45…67.5N, 0…22.5E: the region is its south-west quarter
    assert np.allclose(r[:91, :91], 10) and np.allclose(g[:91, :91], 20) and np.allclose(b[:91, :91], 30)
    assert np.isnan(r[91:]).all() and np.isnan(r[:, 91:]).all()

def test_zoom_seven(tmp_path):
    # One z = 7 tile: 2.8125° square, nodes 1/64° apart, at 47.8125N, 5.625E.
    count = write_tiles(tmp_path, [region(1.0)] * 3, ("node",) * 3, 47.8125, 5.625, zoom=7)
    assert count == 7 and tiles_of(tmp_path)[-1] == "7/49/66.png"
    assert np.allclose(read(tmp_path, "7/49/66.png")[0], 1)

def test_a_zoom_outside_one_to_seven_is_refused(tmp_path):
    with pytest.raises(ValueError, match="zoom 8"):
        write_tiles(tmp_path, [region(1.0)] * 3, ("node",) * 3, 0, 0, zoom=8)

def test_no_data_is_per_channel(tmp_path):
    cape = region(500.0)
    cape[:, :90] = np.nan
    write_tiles(tmp_path, [region(1.0), region(2.0), cape], ("node",) * 3, 45, 0, zoom=5)
    data = (tmp_path / "5/12/16.png").read_bytes()
    pixels = np.asarray(Image.open(io.BytesIO(data)))
    assert (pixels[181, 0] == [1, 1, 0]).all()  # south-west node: R and G hold data, B has none
    r, g, b = decode_tile(data, per_channel=True)
    assert np.allclose(r, 1) and np.allclose(g, 2)
    assert np.isnan(b[:, :90]).all() and np.allclose(b[:, 90:], 500)

def test_only_tiles_with_data_are_written(tmp_path):
    # Two z = 5 tiles wide, the eastern one empty in every channel, the edge they share too.
    field = region(1.0, columns=361)
    field[:, 180:] = np.nan
    write_tiles(tmp_path, [field] * 3, ("mean",) * 3, 45, 0, zoom=5)
    assert [name for name in tiles_of(tmp_path) if name.startswith("5/")] == ["5/12/16.png"]

def test_coarse_zooms_pool_by_channel(tmp_path):
    # One z = 2 tile, 0…90N, 0…90E, nodes 0.5° apart; at z = 1 every 1° node takes ±1 node of 0.5° around it.
    spike, low = region(0.0), region(10.0)
    spike[2, 2] = 90.0  # 1N 1E
    low[3, 3] = 1.0  # 1.5N 1.5E, between the 1° nodes
    write_tiles(tmp_path, [spike, low, spike], ("mean", "min", "max"), 0, 0, zoom=2)
    r, g, b = read(tmp_path, "1/0/1.png")  # 90S…90N, 0…180E: node 1N 1E is row 91, column 1
    assert r[91, 1] == pytest.approx(10, abs=0.2) and r[92, 2] == pytest.approx(0, abs=0.2)
    assert g[91, 1] == pytest.approx(1, abs=0.1) and g[92, 2] == pytest.approx(1, abs=0.1) and g[93, 3] > 9
    assert b[91, 1] == pytest.approx(90, abs=0.2) and b[92, 2] == pytest.approx(0, abs=0.2)
    write_tiles(tmp_path, [spike, low, spike], ("node",) * 3, 0, 0, zoom=2)
    r, g, _ = read(tmp_path, "1/0/1.png")
    assert r[91, 1] == pytest.approx(90, abs=0.2) and g[92, 2] == pytest.approx(10, abs=0.1)

def test_a_global_field_wraps_at_180(tmp_path):
    field = np.zeros((361, 720), dtype=np.float32)  # z = 2: 0.5°, 90S…90N, 180W…179.5E
    field[:, 719] = 50.0  # 179.5E
    count = write_tiles(tmp_path, [field] * 3, ("node", "mean", "max"), -90, -180, zoom=2)
    assert count == 2 + 8
    for name, column in (("1/0/0.png", 0), ("1/0/1.png", 180)):  # 180W…0 and 0…180E share the 180° node
        node, mean, highest = read(tmp_path, name)
        assert node[90, column] == pytest.approx(0, abs=0.2)
        assert mean[90, column] == pytest.approx(50 / 3, abs=0.2)  # 179.5E, 180 and 179.5W
        assert highest[90, column] == pytest.approx(50, abs=0.2)
