from datetime import datetime, timedelta, timezone
import json
import numpy as np
from windtiles.build import build, decide, frames, hazard_fields, level_fields
from windtiles.encode import decode_tile
from windtiles.hazards import pressure_altitude_ft
from windtiles.source import PL_LEVELS

RUN = datetime(2026, 10, 19, 0, tzinfo=timezone.utc)

def globe(value):
    return np.full((721, 1440), value, dtype=np.float32)

def fields(sp_pa=101_325.0):
    f = {("sp", 0, 6): globe(sp_pa), ("cp", 0, 3): globe(0.003), ("cp", 0, 6): globe(0.00765)}  # 1.55 mm/h
    for level in PL_LEVELS:
        isa_m = pressure_altitude_ft(level * 100.0) * 0.3048
        for name, value in (("u", 10.0), ("v", -5.0), ("t", 220.0), ("gh", isa_m), ("r", 50.0)):
            f[(name, level, 6)] = globe(value)
    for name, value in (("10u", 2.0), ("10v", 1.0), ("2t", 288.15), ("mucape", 800.0)):
        f[(name, 0, 6)] = globe(value)
    return f

def test_frames_cover_six_to_sixty_six():
    times = frames(RUN)
    assert len(times) == 21 and times[0].hour == 6 and times[-1] == datetime(2026, 10, 21, 18, tzinfo=timezone.utc)

def test_conversion_to_kt_and_celsius():
    u, v, t = level_fields(fields(), "250", 6)
    assert abs(u[0, 0] - 19.43844) < 1e-4 and abs(v[0, 0] + 9.71922) < 1e-4 and abs(t[0, 0] + 53.15) < 1e-4
    u, _, t = level_fields(fields(), "sfc", 6)
    assert abs(u[0, 0] - 3.887688) < 1e-4 and abs(t[0, 0] - 15) < 1e-4

def test_below_ground_is_masked():
    f = fields(sp_pa=80_000.0)  # ground near 800 hPa
    assert np.isnan(level_fields(f, "850", 6)[0]).all()
    assert not np.isnan(level_fields(f, "700", 6)[0]).any()

def test_surface_level_is_never_masked():
    assert not np.isnan(level_fields(fields(sp_pa=50_000.0), "sfc", 6)[0]).any()

def test_decide():
    now = RUN + timedelta(hours=10)
    assert decide(RUN, None, now) == "build"
    assert decide(RUN, RUN - timedelta(hours=6), now) == "build"
    assert decide(RUN, RUN, now) == "up to date"
    assert decide(RUN, RUN, RUN + timedelta(hours=19)) == "stale"

def test_build_writes_every_tile_and_the_manifest(tmp_path):
    count = build(RUN, fields(), tmp_path, levels=["250"], hazard_levels=[], steps=[6])
    assert count == 42
    assert (tmp_path / "wind/v1/2026101900/2026101906/250/3/3/7.png").exists()
    manifest = json.loads((tmp_path / "wind/v1/manifest.json").read_text())
    assert manifest == {"run": "2026-10-19T00:00:00Z", "frames": ["2026-10-19T06:00:00Z"]}
    assert "ECMWF" in (tmp_path / "index.html").read_text()

def test_hazard_tiles_with_the_wind(tmp_path):
    count = build(RUN, fields(), tmp_path, levels=["250"], hazard_levels=["250", "col"], steps=[6])
    assert count == 126
    # Tile row 2 (0…45°N): row 3 reaches past 85°N, where turbulence and so the whole node has no data.
    r, g, b = decode_tile((tmp_path / "hazards/v1/2026101900/2026101906/250/3/2/7.png").read_bytes())
    # Wind is the same on every level, so no shear; −53 °C, so no icing; 0 °C between the ground (15 °C) and 1000 hPa (−53 °C).
    surface, low = pressure_altitude_ft(np.array([101_325.0, 100_000.0]))
    expected = (surface + (low - surface) * 15 / 68) / 100
    assert np.allclose(r, 0) and np.allclose(g, 0) and np.allclose(b, expected, atol=0.05)
    s, cape, rate = decode_tile((tmp_path / "hazards/v1/2026101900/2026101906/col/3/2/7.png").read_bytes())
    assert np.allclose(s, 0.5, atol=0.01) and np.allclose(cape, 800) and np.allclose(rate, 1.55, atol=0.01)

def test_thunderstorm_zone_cape_and_rate_widen_by_half_a_degree():
    f = fields()
    f[("mucape", 0, 6)], f[("cp", 0, 6)] = globe(0.0), globe(0.003)  # no CAPE, no rain
    f[("mucape", 0, 6)][360, 720], f[("cp", 0, 6)][360, 720] = 1500.0, 0.012  # one node: 3 mm/h
    s, cape, rate = hazard_fields(f, "col", 6)
    for channel, peak in ((s, 1), (cape, 1500), (rate, 3)):  # ±2 nodes = ±0.5° in both axes
        assert np.allclose([channel[360, 720], channel[362, 722], channel[358, 718]], peak, atol=0.01)
        assert channel[363, 720] == 0 and channel[360, 723] == 0 and channel[357, 717] == 0

def test_a_hazard_data_failure_keeps_the_wind(tmp_path, capsys):
    f = fields()
    del f[("mucape", 0, 6)]
    count = build(RUN, f, tmp_path, levels=["250"], hazard_levels=["250", "col"], steps=[6])
    assert count == 42 and len(list((tmp_path / "wind/v1/2026101900/2026101906/250").rglob("*.png"))) == 42
    assert not (tmp_path / "hazards").exists()
    assert (tmp_path / "wind/v1/manifest.json").exists() and (tmp_path / "index.html").exists()
    assert "skip hazards" in capsys.readouterr().out

def test_hazards_below_ground_are_masked():
    s, _, _ = hazard_fields(fields(sp_pa=80_000.0), "850", 6)
    assert np.isnan(s).all()

def test_index_names_the_derived_fields(tmp_path):
    build(RUN, fields(), tmp_path, levels=["250"], hazard_levels=["col"], steps=[6])
    assert "Ellrod" in (tmp_path / "index.html").read_text()
