import math
import numpy as np
import pytest
from PIL import Image
from windtiles.mercator import LEGEND, NO_DATA, write_mercator_tiles

COLOUR = {mm: rgba for mm, rgba in LEGEND}
CLEAR = (0, 0, 0, 0)

def tiles_of(directory):
    return sorted(str(path.relative_to(directory)) for path in directory.rglob("*.png"))

def pixels(directory, name):
    image = Image.open(directory / name)
    assert image.mode == "P" and image.size == (256, 256) and "transparency" in image.info
    return np.asarray(image.convert("RGBA"))

def pixel_of(lat, lon, z):
    """Global (column, row) of the Web Mercator pixel holding lat, lon (standard slippy-map formula)."""
    n = 2**z * 256
    phi = math.radians(lat)
    return int((lon + 180) / 360 * n), int((1 - math.log(math.tan(phi) + 1 / math.cos(phi)) / math.pi) / 2 * n)

def centre_of(column, row, z):
    """lat, lon of the centre of a global pixel."""
    n = 2**z * 256
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (row + 0.5) / n)))), (column + 0.5) / n * 360 - 180

def test_legend_thresholds():
    assert [mm for mm, _ in LEGEND] == [0.5, 1, 2, 5, 10, 15, 20, 30, 50, 75, 100]
    assert all(len(rgba) == 4 and all(0 <= c <= 255 for c in rgba) for _, rgba in LEGEND)

def test_a_region_on_one_tile_fills_exactly_that_tile(tmp_path):
    # z = 2, x = 2, y = 1 is 0…90E, 0…66.5133N; the region reaches to 66.5N, closer than any pixel centre to the top.
    field = np.full((round(66.5 / 0.0625) + 1, round(90 / 0.0625) + 1), 10.0, dtype=np.float32)
    count = write_mercator_tiles(tmp_path, field, 0, 0, 0.0625, max_zoom=2)
    assert tiles_of(tmp_path) == ["0/0/0.png", "1/1/0.png", "2/2/1.png"] and count == 3
    assert (pixels(tmp_path, "2/2/1.png") == COLOUR[10]).all()
    quarter = pixels(tmp_path, "1/1/0.png")  # 0…180E, 0…85.05N: the region is its lower-left quarter, below 66.51N
    assert (quarter[128:, :128] == COLOUR[10]).all() and (quarter[:, 128:] == CLEAR).all()

def test_a_node_lands_in_its_pixel_with_its_colour(tmp_path):
    # ECMWF at z = 5: nodes 0.0625° apart; one node of 12 mm at 52.5N 13.375E, the rest 0.
    field = np.zeros((161, 161), dtype=np.float32)  # 45…55N, 10…20E
    field[120, 54] = 12.0
    write_mercator_tiles(tmp_path, field, 45, 10, 0.0625, max_zoom=5)
    column, row = pixel_of(52.5, 13.375, 5)
    name = f"5/{column // 256}/{row // 256}.png"
    assert [n for n in tiles_of(tmp_path) if n.startswith("5/")] == [name]
    tile = pixels(tmp_path, name)
    assert (tile[row % 256, column % 256] == COLOUR[10]).all()
    painted = np.argwhere(tile[..., 3] > 0)
    for r, c in painted:  # every painted pixel is nearer to this node than to any other
        lat, lon = centre_of(column - column % 256 + c, row - row % 256 + r, 5)
        assert abs(lat - 52.5) <= 0.03125 and abs(lon - 13.375) <= 0.03125

def test_a_dry_field_writes_nothing(tmp_path):
    field = np.full((361, 720), 0.3, dtype=np.float32)
    field[:100] = np.nan
    assert write_mercator_tiles(tmp_path, field, -90, -180, 0.5, max_zoom=3) == 0
    assert tiles_of(tmp_path) == []

def test_no_data_inside_the_data_is_grey_and_around_it_clear(tmp_path):
    # Like a projected grid regridded onto a padded window: data at 40…50N, 0…10E (dry), its north-east quarter
    # 45…50N, 5…10E NaN (a corner of the grid inside its own rectangle), 1.25° of NaN padding all round.
    field = np.full((201, 201), np.nan, dtype=np.float32)  # 38.75…51.25N, 1.25W…11.25E
    field[20:181, 20:181] = 0.2
    field[100:181, 100:181] = np.nan
    count = write_mercator_tiles(tmp_path, field, 38.75, -1.25, 0.0625, max_zoom=5)
    assert count == len(tiles_of(tmp_path)) > 0
    painted = 0
    for name in tiles_of(tmp_path):
        z, x, y = (int(part) for part in name[:-4].split("/"))
        lat = np.array([centre_of(0, y * 256 + r, z)[0] for r in range(256)])
        lon = np.array([centre_of(x * 256 + c, 0, z)[1] for c in range(256)])
        row = np.floor((lat - 38.75) / 0.0625 + 0.5)  # nearest node, as the renderer takes it
        column = np.floor((lon + 1.25) / 0.0625 + 0.5)
        grey = ((row >= 100) & (row <= 180))[:, None] & ((column >= 100) & (column <= 180))
        tile = pixels(tmp_path, name)
        assert (tile[grey] == NO_DATA).all() and (tile[~grey] == CLEAR).all()
        painted += grey.sum()
    assert painted > 0
    assert "5/15/12.png" not in tiles_of(tmp_path)  # dry data and padding only

def test_an_all_nan_field_writes_nothing(tmp_path):
    assert write_mercator_tiles(tmp_path, np.full((161, 161), np.nan), 40, 0, 0.0625, max_zoom=3) == 0

def test_a_global_field_wraps_at_180(tmp_path):
    field = np.zeros((361, 720), dtype=np.float32)  # 0.5°, 90S…90N, 180W…179.5E, the 180°E column dropped
    field[200, 0] = 5.0  # 10N 180W: nearest node for pixels on both sides of the antimeridian
    write_mercator_tiles(tmp_path, field, -90, -180, 0.5, max_zoom=2)
    assert [name for name in tiles_of(tmp_path) if name.startswith("2/")] == ["2/0/1.png", "2/3/1.png"]
    _, row = pixel_of(10, 0, 2)
    assert (pixels(tmp_path, "2/0/1.png")[row % 256, 0] == COLOUR[5]).all()
    assert (pixels(tmp_path, "2/3/1.png")[row % 256, 255] == COLOUR[5]).all()

def test_a_region_crossing_180_paints_both_sides(tmp_path):
    field = np.full((161, 321), 10.0, dtype=np.float32)  # 0…10N, 170E…190E (170W), nodes 1/16° apart
    write_mercator_tiles(tmp_path, field, 0, 170, 0.0625, max_zoom=2)
    assert [name for name in tiles_of(tmp_path) if name.startswith("2/")] == ["2/0/1.png", "2/3/1.png"]
    west, east = pixels(tmp_path, "2/0/1.png"), pixels(tmp_path, "2/3/1.png")
    _, row = pixel_of(5, 0, 2)
    assert (west[row % 256, 0] == COLOUR[10]).all() and (east[row % 256, 255] == COLOUR[10]).all()
    assert (west[row % 256, 30] == CLEAR).all() and (east[row % 256, 200] == CLEAR).all()  # 169.3W, 160.5E

def test_a_region_writes_nothing_outside_its_bbox(tmp_path):
    field = np.full((161, 161), 20.0, dtype=np.float32)  # 40…50N, 0…10E
    write_mercator_tiles(tmp_path, field, 40, 0, 0.0625, max_zoom=5)
    for name in tiles_of(tmp_path):
        z, x, y = (int(part) for part in name[:-4].split("/"))
        lat = np.array([centre_of(0, y * 256 + r, z)[0] for r in range(256)])
        lon = np.array([centre_of(x * 256 + c, 0, z)[1] for c in range(256)])
        inside = ((lat >= 40 - 0.03125) & (lat < 50 + 0.03125))[:, None] & ((lon >= -0.03125) & (lon < 10 + 0.03125))
        tile = pixels(tmp_path, name)
        assert (tile[inside] == COLOUR[20]).all() and (tile[~inside] == CLEAR).all() and inside.any()
    # Pixels within half a node west of 0° take the western nodes: a sliver of column 15.
    fives = [name for name in tiles_of(tmp_path) if name.startswith("5/")]
    assert fives == ["5/15/10.png", "5/15/11.png", "5/15/12.png", "5/16/10.png", "5/16/11.png", "5/16/12.png"]

@pytest.mark.parametrize(
    "mm, colour", [(0.49, None), (0.5, 0.5), (0.99, 0.5), (1, 1), (99.99, 75), (100, 100), (250, 100)]
)
def test_threshold_edges(tmp_path, mm, colour):
    field = np.full((11, 11), mm, dtype=np.float32)
    count = write_mercator_tiles(tmp_path, field, 0, 0, 1.0, max_zoom=0)
    if colour is None:
        assert count == 0
    else:
        tile = pixels(tmp_path, "0/0/0.png")
        assert {tuple(p) for p in tile.reshape(-1, 4)} == {CLEAR, COLOUR[colour]}
