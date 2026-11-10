from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import numpy as np
from PIL import Image
from windtiles import build as build_module, source
from windtiles.build import (
    HAZARD_LEVELS, MANIFEST_URL, NEIGHBORS, STEPS, build, build_sky, decide, frames, hazard_fields, level_fields, main,
    sky_fields,
)
from windtiles.encode import decode_tile
from windtiles.source import CLOUD_MB, ICING_MB, TURBULENCE_MB, WIND_MB

RUN = datetime(2026, 10, 19, 0, tzinfo=timezone.utc)
STEP = "2026101900/2026101906"
SKY_BASE = 42  # sky tiles of `fields()`: the `base` folder alone, it has no cloud fields

def globe(value):
    return np.full((721, 1440), value, dtype=np.float32)

def fields(sp_pa=101_325.0):
    """The same keys `read_fields` gives on a real step: no RH above 600 hPa, no T at 250 hPa."""
    f = {("sp", 0, 6): globe(sp_pa)}
    for mb in {*WIND_MB, *TURBULENCE_MB}:
        f[("u", mb, 6)], f[("v", mb, 6)] = globe(10.0), globe(-5.0)
    for mb in WIND_MB:
        f[("t", mb, 6)] = globe(220.0)
    for mb in ICING_MB:
        f[("r", mb, 6)] = globe(50.0)
    for mb in TURBULENCE_MB:
        f[("gh", mb, 6)] = globe((1000 - mb) * 12.0)
    surface = (("10u", 2.0), ("10v", 1.0), ("2t", 288.15), ("zero", 2590.8), ("cape", 800.0), ("cprat", 1.55 / 3600),
               ("ceiling", 1500.0), ("orog", 300.0), ("vis", 4828.032), ("gust", 10.0))
    for name, value in surface:
        f[(name, 0, 6)] = globe(value)
    return f

def sky(f, cover=40.0):
    """`f` with the sky fields at step 6: `cover` % on every level of the profile, which reaches no cloud top."""
    for name, value in (("tcc", 70.0), ("lcc", 20.0), ("mcc", 40.0), ("hcc", 60.0), ("hpbl", 304.8)):
        f[(name, 0, 6)] = globe(value)
    for mb in CLOUD_MB:
        f[("tcc", mb, 6)] = globe(cover)
        f.setdefault(("gh", mb, 6), globe((1000 - mb) * 12.0))
    return f

def test_frames_cover_three_to_sixty_six():
    times = frames(RUN)
    assert len(times) == 22
    assert times[0] == RUN + timedelta(hours=3) and times[-1] == RUN + timedelta(hours=66)

def test_hazard_levels_and_neighbours():
    assert HAZARD_LEVELS == ["sfc", "900", "850", "800", "750", "700", "600", "500", "400", "300", "col"]
    assert NEIGHBORS == {500: (600, 400), 400: (500, 300), 300: (400, 250)}

def test_manifest_lives_on_the_ga_project():
    assert MANIFEST_URL == "https://aviation-weather-tiles.web.app/wind/v1/manifest.json"

def test_conversion_to_kt_and_celsius():
    u, v, t = level_fields(fields(), "500", 6)
    assert abs(u[0, 0] - 19.43844) < 1e-4 and abs(v[0, 0] + 9.71922) < 1e-4 and abs(t[0, 0] + 53.15) < 1e-4
    u, _, t = level_fields(fields(), "sfc", 6)
    assert abs(u[0, 0] - 3.887688) < 1e-4 and abs(t[0, 0] - 15) < 1e-4

def test_ground_masks_a_level():
    f = fields(sp_pa=84_000.0)  # ground between 850 and 800 hPa
    assert np.isnan(level_fields(f, "850", 6)[0]).all()
    assert not np.isnan(level_fields(f, "800", 6)[0]).any()

def test_surface_level_is_never_masked():
    assert not np.isnan(level_fields(fields(sp_pa=50_000.0), "sfc", 6)[0]).any()

def test_decide():
    now = RUN + timedelta(hours=10)
    assert decide(RUN, None, now) == "build"
    assert decide(RUN, RUN - timedelta(hours=6), now) == "build"
    assert decide(RUN, RUN, now) == "up to date"
    assert decide(RUN, RUN, RUN + timedelta(hours=19)) == "stale"

def test_levels_and_tile_count(tmp_path):
    count, dropped = build(RUN, fields(), tmp_path, steps=[6])
    assert not dropped
    folders = [*(tmp_path / f"wind/v1/{STEP}").iterdir(), *(tmp_path / f"hazards/v1/{STEP}").iterdir()]
    assert len(folders) == 21
    assert all(len(list(folder.rglob("*.png"))) == 42 for folder in folders)
    assert count == 882 + SKY_BASE and 882 * len(STEPS) == 19_404

def test_build_writes_every_tile_and_the_manifest(tmp_path):
    count, _ = build(RUN, fields(), tmp_path, levels=["500"], hazard_levels=[], steps=[6])
    assert count == 42 + SKY_BASE
    assert (tmp_path / f"wind/v1/{STEP}/500/3/3/7.png").exists()
    manifest = json.loads((tmp_path / "wind/v1/manifest.json").read_text())
    assert manifest == {"run": "2026-10-19T00:00:00Z", "frames": ["2026-10-19T06:00:00Z"]}

def test_hazard_tiles_with_the_wind(tmp_path):
    count, _ = build(RUN, fields(), tmp_path, levels=["500"], hazard_levels=["500", "col"], steps=[6])
    assert count == 126 + SKY_BASE
    # Tile row 2 (0…45°N): row 3 reaches past 85°N, where turbulence and so the whole node has no data.
    r, g, b = decode_tile((tmp_path / f"hazards/v1/{STEP}/500/3/2/7.png").read_bytes())
    assert np.allclose(r, 0) and np.allclose(g, 0) and np.allclose(b, 85, atol=0.05)
    s, cape, rate = decode_tile((tmp_path / f"hazards/v1/{STEP}/col/3/2/7.png").read_bytes())
    assert np.allclose(s, 0.5, atol=0.01) and np.allclose(cape, 800) and np.allclose(rate, 1.55, atol=0.01)

def test_no_turbulence_below_500():
    lon, lat = np.meshgrid(np.arange(-180, 180, 0.25), np.arange(-90, 90.25, 0.25))
    x = np.radians(lon) * 6_371_000 * np.cos(np.radians(lat))
    u = (4e-5 * x).astype(np.float32)
    f = fields()
    f[("u", 600, 6)], f[("u", 500, 6)], f[("u", 400, 6)] = u, u, u + 20  # DEF 4e-5, VWS 0.02: moderate
    f[("gh", 600, 6)], f[("gh", 400, 6)] = globe(10_000.0), globe(11_000.0)
    assert (hazard_fields(f, "600", 6)[0] == 0).all()
    assert abs(hazard_fields(f, "500", 6)[0][360, 720] - 0.5) < 0.01

def test_no_icing_above_600():
    f = fields()
    for mb in WIND_MB:
        f[("t", mb, 6)] = globe(265.15)
    for mb in ICING_MB:
        f[("r", mb, 6)] = globe(95.0)
    assert np.allclose(hazard_fields(f, "900", 6)[1], 0.7)
    assert np.allclose(hazard_fields(f, "600", 6)[1], 0.7)
    for level in ("500", "400", "300"):
        assert (hazard_fields(f, level, 6)[1] == 0).all()

def test_surface_conditions_are_ceiling_above_ground_visibility_and_gust(tmp_path):
    f = fields()
    ceiling, visibility, gust = hazard_fields(f, "sfc", 6)
    assert np.allclose(ceiling, 1200 / 0.3048) and np.allclose(visibility, 3) and np.allclose(gust, 19.43844)
    count, dropped = build(RUN, f, tmp_path, levels=["500"], hazard_levels=["sfc"], steps=[6])
    assert not dropped and (tmp_path / "hazards/v1/2026101900/2026101906/sfc/3/2/3.png").exists()


def test_a_fog_pocket_survives_the_coarse_tiles(tmp_path):
    f = fields()
    f[("ceiling", 0, 6)] = globe(19_999.0)
    f[("ceiling", 0, 6)][361, 721] = 400.0  # between the 1° nodes of 0°, 0° and 0.25°N, 0.25°E
    build(RUN, f, tmp_path, levels=["500"], hazard_levels=["sfc"], steps=[6])
    ceiling, _, _ = decode_tile((tmp_path / "hazards/v1/2026101900/2026101906/sfc/1/0/1.png").read_bytes())
    assert ceiling.min() < 12_000


def test_a_broken_surface_field_costs_only_the_surface_tiles(tmp_path):
    f = fields()
    del f[("vis", 0, 6)]
    count, dropped = build(RUN, f, tmp_path, levels=["500"], hazard_levels=["sfc", "500"], steps=[6])
    assert not dropped and not (tmp_path / f"hazards/v1/{STEP}/sfc").exists() and (tmp_path / f"hazards/v1/{STEP}/500").exists()


def test_thunderstorm_reads_cape_and_cprat():
    s, cape, rate = hazard_fields(fields(), "col", 6)
    assert np.allclose(s, 0.5, atol=0.01) and np.allclose(cape, 800) and np.allclose(rate, 1.55, atol=0.01)

def test_thunderstorm_zone_cape_and_rate_widen_by_half_a_degree():
    f = fields()
    f[("cape", 0, 6)], f[("cprat", 0, 6)] = globe(0.0), globe(0.0)  # no CAPE, no rain
    f[("cape", 0, 6)][360, 720], f[("cprat", 0, 6)][360, 720] = 1500.0, 3 / 3600  # one node: 3 mm/h
    s, cape, rate = hazard_fields(f, "col", 6)
    for channel, peak in ((s, 1), (cape, 1500), (rate, 3)):  # ±2 nodes = ±0.5° in both axes
        assert np.allclose([channel[360, 720], channel[362, 722], channel[358, 718]], peak, atol=0.01)
        assert channel[363, 720] == 0 and channel[360, 723] == 0 and channel[357, 717] == 0

def test_a_hazard_failure_keeps_the_wind(tmp_path, capsys):
    f = fields()
    del f[("zero", 0, 6)]
    # col needs no freezing level: its tiles are written, and the failure at 500 has them to clean up.
    count, dropped = build(RUN, f, tmp_path, levels=["500"], hazard_levels=["col", "500"], steps=[6])
    assert dropped and count == 42 + SKY_BASE and len(list((tmp_path / f"wind/v1/{STEP}/500").rglob("*.png"))) == 42
    assert not (tmp_path / "hazards").exists()
    assert (tmp_path / "wind/v1/manifest.json").exists() and (tmp_path / "index.html").exists()
    assert "skip hazards" in capsys.readouterr().out

def test_hazards_below_ground_are_masked():
    f = fields(sp_pa=80_000.0)
    assert all(np.isnan(channel).all() for channel in hazard_fields(f, "850", 6))

def test_index_credits_noaa_and_names_the_derived_fields(tmp_path):
    build(RUN, fields(), tmp_path, levels=["500"], hazard_levels=["col"], steps=[6])
    html = (tmp_path / "index.html").read_text()
    assert "Wind tiles for Aviation Weather. Based on NOAA GFS 0.25° data, public domain." in html
    assert "NOAA does not endorse this service." in html
    assert "Ellrod" in html
    assert "ECMWF" not in html and "Pilot Force" not in html

def stub_source(monkeypatch, run, deployed=None, dropped=False):
    """main() on stubs; returns the calls to `download` and `build`."""
    calls = []
    monkeypatch.setattr(source, "latest_run", lambda now, exists: run)
    monkeypatch.setattr(build_module, "deployed_run", lambda: deployed)
    monkeypatch.setattr(source, "download", lambda *args: calls.append(("download", args)) or ["f.grib2"])
    monkeypatch.setattr(source, "read_fields", lambda paths: {"paths": paths})
    monkeypatch.setattr(build_module, "build", lambda *args: calls.append(("build", args)) or (0, dropped))
    return calls

def test_main_builds_the_latest_run(tmp_path, monkeypatch):
    calls = stub_source(monkeypatch, RUN)
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert main(["--force", "--out", str(tmp_path / "site"), "--work", str(tmp_path / "grib")]) == 0
    assert calls == [
        ("download", (RUN, STEPS, tmp_path / "grib")),
        ("build", (RUN, {"paths": ["f.grib2"]}, tmp_path / "site")),
    ]
    assert output.read_text() == "changed=true\n"

def test_main_without_a_run_builds_nothing(tmp_path, monkeypatch, capsys):
    calls = stub_source(monkeypatch, None)
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert main(["--force", "--out", str(tmp_path / "site"), "--work", str(tmp_path / "grib")]) == 1
    assert calls == [] and output.read_text() == "changed=false\n"
    assert "no GFS run with step 66" in capsys.readouterr().out

def run_with_dropped_hazards(tmp_path, monkeypatch, deployed_age, *flags):
    """main() on a run just out, its hazards dropped, the deployed run `deployed_age` old (`None`: nothing deployed)."""
    latest = datetime.now(timezone.utc)
    deployed = None if deployed_age is None else latest - deployed_age
    stub_source(monkeypatch, latest, deployed=deployed, dropped=True)
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    return main([*flags, "--out", str(tmp_path / "site"), "--work", str(tmp_path / "grib")]), output.read_text()

def test_dropped_hazards_keep_a_young_deployed_run(tmp_path, monkeypatch, capsys):
    assert run_with_dropped_hazards(tmp_path, monkeypatch, timedelta(hours=6)) == (1, "changed=false\n")
    assert "::error" in capsys.readouterr().out

def test_dropped_hazards_keep_a_young_deployed_run_under_force(tmp_path, monkeypatch, capsys):
    assert run_with_dropped_hazards(tmp_path, monkeypatch, timedelta(hours=6), "--force") == (1, "changed=false\n")
    assert "::error" in capsys.readouterr().out

def test_dropped_hazards_deploy_the_wind_over_an_old_run(tmp_path, monkeypatch, capsys):
    assert run_with_dropped_hazards(tmp_path, monkeypatch, timedelta(hours=19)) == (0, "changed=true\n")
    assert "::error" in capsys.readouterr().out

def test_dropped_hazards_deploy_the_wind_with_nothing_deployed(tmp_path, monkeypatch, capsys):
    assert run_with_dropped_hazards(tmp_path, monkeypatch, None) == (0, "changed=true\n")
    assert "::error" in capsys.readouterr().out


def test_sky_fields_and_their_units():
    ceiling, visibility, cape = sky_fields(sky(fields()), "base", 6)
    assert np.allclose(ceiling, 1200 / 0.3048) and np.allclose(visibility, 3) and np.allclose(cape, 800)
    assert [float(x[0, 0]) for x in sky_fields(sky(fields()), "cloud", 6)] == [20, 40, 60]
    total, thermals, top = sky_fields(sky(fields()), "sky", 6)
    assert np.allclose(total, 70) and np.allclose(thermals, 1000) and np.isnan(top).all()

def test_cloud_top_is_the_highest_cloudy_level_above_ground():
    f = sky(fields(), cover=0.0)
    f[("tcc", 850, 6)] = globe(80.0)
    f[("tcc", 850, 6)][0, 0] = 49.0  # less than half: no cloud top
    f[("tcc", 300, 6)][1, 1] = 50.0
    f[("tcc", 1000, 6)][2, 2] = 90.0
    f[("sp", 0, 6)][2, 2] = 95_000.0  # 1000 hPa under the ground there
    f[("tcc", 1000, 6)][3, 3] = 90.0  # 1000 hPa above the ground, but 850 is higher
    top = sky_fields(f, "sky", 6)[2]
    assert np.isnan(top[0, 0])
    assert np.isclose(top[1, 1], 8400 / 0.3048) and np.allclose([top[2, 2], top[3, 3], top[4, 4]], 1800 / 0.3048)

def test_sky_tiles_manifest_and_models(tmp_path):
    count = build_sky(RUN, sky(fields()), tmp_path, steps=[6])
    assert count == 3 * 42
    for folder in ("base", "cloud", "sky"):
        assert len(list((tmp_path / f"sky/v1/gfs/{STEP}/{folder}").rglob("*.png"))) == 42
    assert json.loads((tmp_path / "sky/v1/gfs/manifest.json").read_text()) == {
        "run": "2026-10-19T00:00:00Z", "frames": ["2026-10-19T06:00:00Z"],
    }
    assert json.loads((tmp_path / "sky/v1/models.json").read_text()) == [{
        "id": "gfs", "title": "GFS", "km": 25, "bbox": [-90, -180, 90, 180], "zoom": 3,
        "layers": ["clouds", "lowClouds", "mediumClouds", "highClouds", "cloudBase", "cloudTops", "visibility", "fog",
                   "cape", "thermals"],
    }]
    data = (tmp_path / f"sky/v1/gfs/{STEP}/sky/3/2/7.png").read_bytes()
    assert (np.asarray(Image.open(io.BytesIO(data)))[1:, :, 2] == 0).all()  # no cloud top: byte 0 in B alone
    total, thermals, top = decode_tile(data, per_channel=True)
    assert np.allclose(total, 70) and np.allclose(thermals, 1000) and np.isnan(top).all()

def test_a_missing_sky_field_costs_its_folder_alone(tmp_path, capsys):
    f = sky(fields())
    del f[("hpbl", 0, 6)]
    count, dropped = build(RUN, f, tmp_path, levels=["500"], hazard_levels=["col"], steps=[6])
    assert not dropped and count == 42 + 42 + 2 * 42
    assert not (tmp_path / f"sky/v1/gfs/{STEP}/sky").exists() and (tmp_path / f"sky/v1/gfs/{STEP}/cloud").exists()
    assert "skip sky sky at step 6" in capsys.readouterr().out

def regression_fields():
    """`sky(fields())` with noise, ground across 850 hPa, a band without a ceiling and a gap in CAPE. Wind and heights
    stay flat at 600…400 hPa: turbulence is then 0 everywhere, so no trigonometry enters the tiles and the digest holds
    on any platform."""
    rng = np.random.default_rng(7)
    f = sky(fields())
    for key, value in f.items():
        if not (key[0] in ("u", "v", "gh") and key[1] in TURBULENCE_MB):
            f[key] = (value * (1 + 0.2 * rng.standard_normal(value.shape))).astype(np.float32)
    f[("t", 850, 6)] = (265 + 3 * rng.standard_normal((721, 1440))).astype(np.float32)
    f[("r", 850, 6)] = rng.uniform(60, 100, (721, 1440)).astype(np.float32)
    f[("sp", 0, 6)] = rng.uniform(50_000, 105_000, (721, 1440)).astype(np.float32)
    f[("ceiling", 0, 6)][:, :400] = 19_999.0
    f[("cape", 0, 6)][300:310, 700:710] = np.nan
    return f

def test_wind_and_hazard_tiles_are_unchanged_by_the_sky_family(tmp_path):
    """The digest of the pixels and paths of every wind and hazard tile, made by the code before the sky family
    (commit 1cabc7c) from the same fields."""
    build(RUN, regression_fields(), tmp_path, levels=["sfc", "850", "500"], hazard_levels=["sfc", "850", "500", "col"],
          steps=[6])
    digest = hashlib.sha256()
    paths = sorted(p for family in ("wind", "hazards") for p in (tmp_path / family).rglob("*.png"))
    for path in paths:
        digest.update(str(path.relative_to(tmp_path)).encode() + np.asarray(Image.open(path)).tobytes())
    assert len(paths) == 7 * 42 and (tmp_path / "sky/v1/gfs/manifest.json").exists()
    assert digest.hexdigest() == "25def717feb3c1ae4ac8d51960ce1ee5d38519cdc56cc22ae97556f73c9bf055"

def test_main_builds_the_sky_alone(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(source, "latest_run", lambda now, exists: RUN)
    monkeypatch.setattr(build_module, "deployed_run", lambda: None)
    monkeypatch.setattr(source, "download", lambda *args, fields: calls.append(("download", *args, fields)) or ["f"])
    monkeypatch.setattr(source, "read_fields", lambda paths: {"paths": paths})
    monkeypatch.setattr(build_module, "build_sky", lambda *args: calls.append(("build_sky", *args)) or 0)
    monkeypatch.setattr(build_module, "build", lambda *args: calls.append(("build", *args)))
    assert main(["--force", "--sky-only", "--out", str(tmp_path / "site"), "--work", str(tmp_path / "grib")]) == 0
    assert calls == [
        ("download", RUN, STEPS, tmp_path / "grib", source.SKY_FIELDS),
        ("build_sky", RUN, {"paths": ["f"]}, tmp_path / "site"),
    ]
