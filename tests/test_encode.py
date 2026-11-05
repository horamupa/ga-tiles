import io, struct
import numpy as np
from PIL import Image
from windtiles.encode import decode_tile, encode_tile
from windtiles.fixture import fixture_fields

def test_round_trip_within_half_a_step():
    u, v, t = fixture_fields()
    du, dv, dt = decode_tile(encode_tile(u, v, t))
    ok = ~np.isnan(u)
    assert np.nanmax(abs(du - u)[ok]) <= 180 / 508 + 1e-6
    assert np.nanmax(abs(dv - v)[ok]) <= 180 / 508 + 1e-6
    assert np.nanmax(abs(dt - t)[ok]) <= 20 / 508 + 1e-6

def test_header_is_six_little_endian_floats():
    u, v, t = fixture_fields()
    image = Image.open(io.BytesIO(encode_tile(u, v, t)))
    assert image.mode == "RGB" and image.size == (181, 182)
    header = bytes(np.asarray(image)[0, :8].reshape(-1))
    assert struct.unpack("<6f", header) == (-90, 90, -90, 90, -60, -40)

def test_no_data_is_byte_zero():
    u, v, t = fixture_fields()
    pixels = np.asarray(Image.open(io.BytesIO(encode_tile(u, v, t))))
    assert (pixels[181, 0] == 0).all()        # node j = 0, i = 0
    assert (pixels[181 - 10, 10] > 0).all()   # node j = 10, i = 10

def test_top_data_row_is_north():
    u, v, t = fixture_fields()
    pixels = np.asarray(Image.open(io.BytesIO(encode_tile(u, v, t))))
    assert pixels[1, 50, 1] == 255 and pixels[181, 50, 1] == 1  # v: north +90, south −90

def test_a_flat_field_decodes_to_its_value():
    flat = np.full((181, 181), 12.5, dtype=np.float32)
    du, _, _ = decode_tile(encode_tile(flat, flat, flat))
    assert np.all(du == 12.5)

def test_no_colour_chunks():
    u, v, t = fixture_fields()
    info = Image.open(io.BytesIO(encode_tile(u, v, t))).info
    assert not {"icc_profile", "srgb", "gamma"} & set(info)
