import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from windtiles import build_models, models
from windtiles.build_models import build_model, clean_source, main, sky_channels, steps, variables
from windtiles.encode import decode_tile
from windtiles.om_source import Fields, Run

RUN = datetime(2026, 11, 10, 12, tzinfo=timezone.utc)
BY_ID = {model.id: model for model in models.MODELS}
FRAME = "sky/v1/ukv/2026111012/2026111013"


def run(hours) -> Run:
    return Run(RUN, [RUN + timedelta(hours=h) for h in hours], [], RUN)


def test_registry_has_twelve_models_in_order_of_preference_with_their_layers():
    assert [m.id for m in models.MODELS] == [
        "hrrr", "hrdps", "msm", "ukv", "aromeHd", "arome", "iconD2", "aladin", "iconEu", "ecmwf", "icon", "gfs",
    ]
    for model in models.MODELS:
        south, west, north, east = model.bbox
        assert -90 <= south < north <= 90 and -180 <= west < east <= 180 and 1 <= model.zoom <= 7
        if model.home:  # inside the bbox, but east may pass 180
            home_south, home_west, home_north, home_east = model.home
            assert south <= home_south < home_north <= north and west <= home_west < home_east <= east + 360
        assert set(model.layers) <= set(models.CLOUD_LAYERS)
        assert ("visibility" in model.layers) == ("fog" in model.layers)  # fog is drawn from visibility
        assert model.source == "gfs" or "cloudTops" not in model.layers  # no bucket model has cloud tops
    assert "cloudBase" in BY_ID["ukv"].layers and "thermals" not in BY_ID["ukv"].layers
    assert "clouds" not in BY_ID["aromeHd"].layers and "thermals" in BY_ID["hrrr"].layers
    assert variables(BY_ID["ecmwf"]) == {
        "cloud_cover": "%", "cloud_cover_low": "%", "cloud_cover_mid": "%", "cloud_cover_high": "%",
        "visibility": "m", "cape": "J/kg", "boundary_layer_height": "m",
    }
    assert list(variables(BY_ID["icon"])) == ["cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
                                              "cape"]


def test_models_json_lists_the_models_the_site_holds(tmp_path):
    assert json.loads(models.models_json(tmp_path)) == []
    for model_id in ("gfs", "iconD2", "ecmwf", "hrrr", "ukv"):
        (tmp_path / f"sky/v1/{model_id}").mkdir(parents=True)
        (tmp_path / f"sky/v1/{model_id}/manifest.json").write_text("{}")
    (tmp_path / "sky/v1/arome/2026111012").mkdir(parents=True)  # tiles without a manifest: not listed
    listed = json.loads(models.models_json(tmp_path))
    assert [m["id"] for m in listed] == ["hrrr", "ukv", "iconD2", "ecmwf", "gfs"]  # registry order
    assert listed[2] == {  # no home: the whole bbox
        "id": "iconD2", "title": "ICON-D2", "km": 2.2, "bbox": [43.18, -3.94, 58.08, 20.34], "zoom": 7,
        "layers": ["clouds", "lowClouds", "mediumClouds", "highClouds", "visibility", "fog", "cape"],
    }
    assert [m.get("home") for m in listed] == [None, [49, -11, 61, 2.5], None, [-90, -30, 90, 190], None]
    assert BY_ID["arome"].home == BY_ID["aromeHd"].home == (37.5, -12, 51.5, 8.5)
    assert BY_ID["icon"].home == (-90, -30, 90, 190)


def test_frames_of_global_and_regional_models():
    assert steps(BY_ID["icon"], run(range(181))) == list(range(3, 67, 3))
    assert steps(BY_ID["ecmwf"], run([*range(0, 90), *range(90, 145, 3)])) == list(range(3, 67, 3))
    assert steps(BY_ID["iconD2"], run(range(49))) == list(range(1, 49))
    assert steps(BY_ID["ukv"], run(range(55))) == list(range(1, 49))
    assert steps(BY_ID["hrrr"], run(range(19))) == list(range(1, 19))  # a run that ends early stops there
    assert steps(BY_ID["msm"], run(range(79))) == list(range(1, 49))


def test_msm_builds_from_the_runs_that_reach_48_hours(tmp_path):
    hours = []
    read, _ = fake_read()
    build_model(BY_ID["msm"]._replace(zoom=3), tmp_path, latest_run=lambda source, h: hours.append(h) or run([1]),
                read=read)
    assert hours == [(0, 12)] and BY_ID["hrdps"].runs == (0, 6, 12, 18)


def channels(model_id, **values):
    shape = (2, 3)
    return sky_channels(BY_ID[model_id], {k: np.full(shape, v, np.float32) for k, v in values.items()})


def test_units_of_every_channel():
    c = channels("hrrr", cloud_cover=70, cloud_cover_low=20, cloud_cover_mid=40, cloud_cover_high=60, cape=800,
                 visibility=4 * 1609.344, boundary_layer_height=304.8)
    ceiling, visibility, cape = c["base"]
    assert np.isnan(ceiling).all() and np.allclose(visibility, 4) and np.allclose(cape, 800)
    assert [float(x[0, 0]) for x in c["cloud"]] == [20, 40, 60]
    total, thermals, top = c["sky"]
    assert np.allclose(total, 70) and np.allclose(thermals, 1000) and np.isnan(top).all()
    assert np.allclose(channels("hrrr", visibility=50_000)["base"][1], 10)


def test_ukv_cloud_base_in_feet_and_none_is_the_cap():
    base = np.array([[np.inf, 0, 304.8, 3657.6, 9000, np.nan]], np.float32)
    ceiling = sky_channels(BY_ID["ukv"], {"cloud_base": base})["base"][0]
    np.testing.assert_allclose(ceiling[0, :5], [12_000, 0, 1000, 12_000, 12_000], rtol=1e-6)
    assert np.isnan(ceiling[0, 5])  # outside the domain


def test_ukv_zero_base_is_fog_under_low_cloud_else_none():
    source = {"cloud_base": np.array([0, 0, 0, 300], np.float32),
              "cloud_cover_low": np.array([49, 50, 100, 0], np.float32)}
    clean_source(source)
    np.testing.assert_array_equal(source["cloud_base"], [np.inf, 0, 0, 300])


def test_negative_cape_is_no_data():
    source = {"cape": np.array([-1000, 0, 250], np.float32)}
    clean_source(source)
    assert np.isnan(source["cape"][0]) and list(source["cape"][1:]) == [0, 250]


def test_aladin_visibility_from_its_cap_is_unlimited():
    metres = [7_399, 7_400, 7_480]
    aladin = sky_channels(BY_ID["aladin"], {"visibility": np.array([metres], np.float32)})["base"][1][0]
    icon_d2 = sky_channels(BY_ID["iconD2"], {"visibility": np.array([metres], np.float32)})["base"][1][0]
    np.testing.assert_allclose(aladin, [7_399 / 1609.344, 10, 10], rtol=1e-6)
    np.testing.assert_allclose(icon_d2, np.array(metres) / 1609.344, rtol=1e-6)


def fake_read(fail_hours=(), south=45.0, west=0.0, shape=(181, 181)):
    """A reader of uniform fields on a zoom-3 window, which fails at `fail_hours` after the run."""
    calls = []

    def read(source, run_time, valid, names, zoom, prepare):
        calls.append((source, valid, tuple(names), zoom, prepare))
        if (valid - run_time) / timedelta(hours=1) in fail_hours:
            raise OSError("range request failed")
        values = {name: np.full(shape, value, np.float32) for name, value in (
            ("cloud_cover", 90), ("cloud_cover_low", 80), ("cloud_cover_mid", 50), ("cloud_cover_high", 10),
            ("visibility", 1609.344), ("cape", 300), ("cloud_base", 609.6),
        ) if name in names}
        units = {name: unit for name, unit in variables(BY_ID["ukv"]).items() if name in names}
        return Fields(south, west, values, units, 1000)

    return read, calls


def test_a_frame_through_write_tiles(tmp_path):
    model = BY_ID["ukv"]._replace(zoom=3)
    read, calls = fake_read()
    result = build_model(model, tmp_path, workers=2, latest_run=lambda source, hours: run([0, 1, 2]), read=read)
    assert result == (RUN, 2, 2 * 3 * 3, 2000)  # 2 frames × 3 folders × a tile at each of zooms 1…3
    assert calls[0] == ("ukmo_uk_deterministic_2km", RUN + timedelta(hours=1),
                        ("cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "cloud_base",
                         "visibility", "cape"), 3, clean_source)
    # 45…90°N, 0…45°E at zoom 3 is tile row 3, column 4; the sky folder has total cover alone.
    ceiling, visibility, cape = decode_tile((tmp_path / f"{FRAME}/base/3/3/4.png").read_bytes(), per_channel=True)
    assert np.allclose(ceiling, 2000, atol=1) and np.allclose(visibility, 1) and np.allclose(cape, 300)
    total, thermals, top = decode_tile((tmp_path / f"{FRAME}/sky/3/3/4.png").read_bytes(), per_channel=True)
    assert np.allclose(total, 90) and np.isnan(thermals).all() and np.isnan(top).all()
    assert sorted(p.relative_to(tmp_path / FRAME).as_posix() for p in (tmp_path / FRAME).rglob("*.png")) == [
        f"{folder}/{z}" for folder in ("base", "cloud", "sky") for z in ("1/0/1.png", "2/1/2.png", "3/3/4.png")
    ]
    assert json.loads((tmp_path / "sky/v1/ukv/manifest.json").read_text()) == {
        "run": "2026-11-10T12:00:00Z", "frames": ["2026-11-10T13:00:00Z", "2026-11-10T14:00:00Z"],
    }


def test_a_folder_without_channels_is_not_written(tmp_path):
    model = BY_ID["aromeHd"]._replace(zoom=3)
    read, _ = fake_read()
    build_model(model, tmp_path, latest_run=lambda source, hours: run([1]), read=read)
    frame = tmp_path / "sky/v1/aromeHd/2026111012/2026111013"
    assert sorted(p.name for p in frame.iterdir()) == ["base", "cloud"]


def test_a_global_field_wraps_at_the_date_line(tmp_path):
    model = BY_ID["icon"]._replace(zoom=1)
    lon = np.tile(np.arange(-180, 181, dtype=np.float32), (181, 1))  # 361 columns: the last is 180°E again

    def read(source, run_time, valid, names, zoom, prepare):
        return Fields(-90, -180, {name: lon for name in names}, variables(model), 0)

    build_model(model, tmp_path, latest_run=lambda source, hours: run([3]), read=read)
    low = decode_tile((tmp_path / "sky/v1/icon/2026111012/2026111015/cloud/1/0/1.png").read_bytes(), True)[0]
    assert low[90, -1] == pytest.approx(-180, abs=1)  # 180°E reads column 0, not the duplicate


def test_a_failed_frame_is_left_out_and_the_manifest_comes_last(tmp_path, capsys):
    model = BY_ID["ukv"]._replace(zoom=3)
    read, _ = fake_read(fail_hours=(2,))
    old = tmp_path / "sky/v1/ukv/manifest.json"
    assert build_model(model, tmp_path, latest_run=lambda source, hours: run([0, 1, 2, 3]), read=read)[1] == 2
    assert json.loads(old.read_text())["frames"] == ["2026-11-10T13:00:00Z", "2026-11-10T15:00:00Z"]
    assert "ukv: skip frame +2 h: OSError('range request failed')" in capsys.readouterr().out

    read, _ = fake_read(fail_hours=(1, 2, 3))
    with pytest.raises(RuntimeError):
        build_model(model, tmp_path, latest_run=lambda source, hours: run([0, 1, 2, 3]), read=read)
    assert json.loads(old.read_text())["frames"] == ["2026-11-10T13:00:00Z", "2026-11-10T15:00:00Z"]  # untouched
    assert (tmp_path / f"{FRAME}/base/3/3/4.png").exists()  # and its tiles are still there


def test_wrong_units_fail_the_frame(tmp_path):
    def read(source, run_time, valid, names, zoom, prepare):
        fields = fake_read()[0](source, run_time, valid, names, zoom, prepare)
        return fields._replace(units={**fields.units, "visibility": "km"})

    with pytest.raises(RuntimeError):
        build_model(BY_ID["ukv"]._replace(zoom=3), tmp_path, latest_run=lambda source, hours: run([1]), read=read)


def test_main_writes_models_json_of_the_site_after_a_failed_model(tmp_path, monkeypatch, capsys):
    (tmp_path / "sky/v1/gfs").mkdir(parents=True)
    (tmp_path / "sky/v1/gfs/manifest.json").write_text("{}")

    def build(model, out, workers, deployed):
        if model.id == "hrrr":
            raise RuntimeError("bucket down")
        (out / f"sky/v1/{model.id}").mkdir(parents=True)
        (out / f"sky/v1/{model.id}/manifest.json").write_text("{}")
        return RUN, 48, 100, 5_000_000

    monkeypatch.setattr(build_models, "build_model", build)
    assert main(["--out", str(tmp_path), "--models", "hrrr", "iconEu"]) == 1
    assert [m["id"] for m in json.loads((tmp_path / "sky/v1/models.json").read_text())] == ["iconEu", "gfs"]
    out = capsys.readouterr().out
    assert "::error::hrrr: RuntimeError('bucket down')" in out and "iconEu: run 2026-11-10T12:00:00Z, 48 frames" in out


def test_a_deployed_run_is_skipped(tmp_path, monkeypatch, capsys):
    read, calls = fake_read()
    assert build_model(BY_ID["ukv"], tmp_path, latest_run=lambda source, h: run([1]), read=read, deployed=RUN) is None
    assert calls == [] and not (tmp_path / "sky").exists()
    assert "ukv: run 2026-11-10T12:00:00Z up to date" in capsys.readouterr().out

    deployed = tmp_path / "deployed"  # the manifests pulled from the bucket
    (deployed / "sky/v1/ukv").mkdir(parents=True)
    (deployed / "sky/v1/ukv/manifest.json").write_text('{"run": "2026-11-10T12:00:00Z", "frames": []}')
    got = {}

    def build(model, out, workers, deployed):
        got[model.id] = deployed

    monkeypatch.setattr(build_models, "build_model", build)
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert main(["--out", str(tmp_path / "site"), "--models", "ukv", "hrrr", "--deployed", deployed.as_uri()]) == 0
    assert got == {"hrrr": None, "ukv": RUN} and output.read_text() == "changed=false\n"


def test_the_models_workflow_builds_every_bucket_model():
    workflow = (Path(__file__).parents[1] / ".github/workflows/models.yml").read_text()
    matrix = re.search(r"^\s+model: \[(.*)\]$", workflow, re.MULTILINE).group(1)
    assert matrix.split(", ") == [model.id for model in models.MODELS if model.source != "gfs"]
