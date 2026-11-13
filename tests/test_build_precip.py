import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from PIL import Image

from windtiles import build_precip, models
from windtiles.build_precip import build_model, index_json, main, window_start
from windtiles.mercator import LEGEND
from windtiles.om_source import Fields, Run

RUN = datetime(2026, 11, 12, 6, tzinfo=timezone.utc)
NOW = datetime(2026, 11, 12, 10, 37, tzinfo=timezone.utc)  # start 10Z, 4 h after the run
BY_ID = {model.id: model for model in models.MODELS}
COLOUR = {mm: rgba for mm, rgba in LEGEND}
MODEL = BY_ID["iconD2"]._replace(zoom=3)  # zoom 3: 1/4° nodes, tiles of zooms 0…3


def run(hours) -> Run:
    return Run(RUN, [RUN + timedelta(hours=h) for h in hours], [], RUN)


def fake_read(mm=lambda hour: 1.0, fail_hours=(), units=lambda hour: "mm", south=46.0, west=1.0, shape=(41, 41)):
    """A reader of a uniform `mm(hour)` field per frame, hours after the run; fails at `fail_hours`."""
    calls = []

    def read(source, run_time, valid, names, zoom):
        hour = round((valid - run_time) / timedelta(hours=1))
        calls.append((source, valid, tuple(names), zoom))
        if hour in fail_hours:
            raise OSError("range request failed")
        values = {"precipitation": np.full(shape, mm(hour), np.float32)}
        return Fields(south, west, values, {"precipitation": units(hour)}, 1000)

    return read, calls


def colours(path):
    """The colours of a tile's painted pixels."""
    image = Image.open(path)
    assert image.mode == "P" and "transparency" in image.info
    rgba = np.asarray(image.convert("RGBA")).reshape(-1, 4)
    return {tuple(int(c) for c in pixel) for pixel in rgba if pixel[3]}


def test_start_is_the_hour_floor_and_never_before_the_run():
    assert window_start(RUN, NOW) == datetime(2026, 11, 12, 10, tzinfo=timezone.utc)
    assert window_start(RUN, datetime(2026, 11, 12, 5, 59, tzinfo=timezone.utc)) == RUN
    assert window_start(RUN, datetime(2026, 11, 12, 6, 0, 1, tzinfo=timezone.utc)) == RUN


def test_window_sums_over_the_hours_after_the_start(tmp_path):
    # 0.1 mm an hour for the first 12 h after the start, then 2 mm: the sums are 1.2, 25.2 and 73.2 mm.
    read, calls = fake_read(mm=lambda hour: 0.1 if hour <= 4 + 12 else 2.0)
    result = build_model(MODEL, tmp_path, workers=3, latest_run=lambda source, hours: run(range(1, 79)), read=read,
                         now=NOW)
    start = datetime(2026, 11, 12, 10, tzinfo=timezone.utc)
    assert result == (RUN, start, ["12h", "24h", "48h"], 3 * 4, 48 * 1000)  # a tile at each of zooms 0…3
    assert sorted(valid for _, valid, _, _ in calls) == [start + timedelta(hours=h) for h in range(1, 49)]
    assert calls[0][0] == "dwd_icon_d2" and calls[0][2:] == (("precipitation",), 3)
    frame = tmp_path / "precip/v1/iconD2/2026111206/2026111210"
    # 46…56°N, 1…11°E (half a node more around) at zoom 3 is in tile x 4, y 2.
    assert colours(frame / "12h/3/4/2.png") == {COLOUR[1]}
    assert colours(frame / "24h/3/4/2.png") == {COLOUR[20]}
    assert colours(frame / "48h/3/4/2.png") == {COLOUR[50]}
    assert sorted(p.relative_to(frame / "12h").as_posix() for p in (frame / "12h").rglob("*.png")) == [
        "0/0/0.png", "1/1/0.png", "2/2/1.png", "3/4/2.png",
    ]
    assert json.loads((tmp_path / "precip/v1/iconD2/manifest.json").read_text()) == {
        "run": "2026-11-12T06:00Z", "start": "2026-11-12T10:00Z", "windows": ["12h", "24h", "48h"], "maxZoom": 3,
    }


def test_a_regional_run_ending_at_48_hours_publishes_the_windows_it_completes(tmp_path):
    read, calls = fake_read()
    result = build_model(MODEL, tmp_path, latest_run=lambda source, hours: run(range(1, 49)), read=read, now=NOW)
    assert result[2] == ["12h", "24h"] and len(calls) == 24  # +44 h from the start: no 48 h, nor its frames
    frame = tmp_path / "precip/v1/iconD2/2026111206/2026111210"
    assert sorted(p.name for p in frame.iterdir()) == ["12h", "24h"]


def test_a_missing_hour_drops_only_the_windows_that_cover_it(tmp_path, capsys):
    read, _ = fake_read(fail_hours=(4 + 13,))  # 13 h after the start
    result = build_model(MODEL, tmp_path, latest_run=lambda source, hours: run(range(1, 79)), read=read, now=NOW)
    assert result[2] == ["12h"] and result[4] == 47 * 1000
    assert "iconD2: skip frame +13 h from 2026-11-12T10:00Z: OSError('range request failed')" in capsys.readouterr().out

    read, _ = fake_read()  # a run without hour 30 after the start
    result = build_model(MODEL, tmp_path, latest_run=lambda s, h: run([*range(1, 34), *range(35, 79)]), read=read,
                         now=NOW)
    assert result[2] == ["12h", "24h"]


def test_a_wrong_unit_is_a_missing_hour(tmp_path, capsys):
    read, _ = fake_read(units=lambda hour: "kg m-2" if hour == 4 + 20 else "mm")
    assert build_model(MODEL, tmp_path, latest_run=lambda s, h: run(range(1, 79)), read=read, now=NOW)[2] == ["12h"]
    assert "ValueError(\"unit 'kg m-2', expected 'mm'\")" in capsys.readouterr().out


def test_no_complete_window_fails_and_the_old_manifest_stays(tmp_path):
    manifest = tmp_path / "precip/v1/iconD2/manifest.json"
    read, _ = fake_read()
    build_model(MODEL, tmp_path, latest_run=lambda s, h: run(range(1, 79)), read=read, now=NOW)
    old = manifest.read_text()
    later = NOW + timedelta(hours=1)
    read, _ = fake_read(fail_hours=(5 + 1,))
    with pytest.raises(RuntimeError, match="no complete window"):
        build_model(MODEL, tmp_path, latest_run=lambda s, h: run(range(1, 79)), read=read, now=later)
    with pytest.raises(RuntimeError, match="has no window"):  # a run that ends within 12 h of the start
        build_model(MODEL, tmp_path, latest_run=lambda s, h: run(range(1, 15)), read=read, now=later)
    assert manifest.read_text() == old


def test_a_dry_window_is_complete_without_tiles(tmp_path):
    read, _ = fake_read(mm=lambda hour: 0.0)
    result = build_model(MODEL, tmp_path, latest_run=lambda s, h: run(range(1, 79)), read=read, now=NOW)
    assert result[2:4] == (["12h", "24h", "48h"], 0) and not list(tmp_path.rglob("*.png"))


def test_two_runs_with_the_same_start_write_separate_folders(tmp_path):
    deployed = {"run": "2026-11-12T00:00Z", "start": "2026-11-12T10:00Z", "windows": ["12h"], "maxZoom": 3}
    old = tmp_path / "precip/v1/iconD2/2026111200/2026111210/12h/3/4/2.png"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"wet in the 00Z run")
    read, _ = fake_read()  # the 06Z run of the same start hour is not skipped and does not touch the 00Z folder
    result = build_model(MODEL, tmp_path, latest_run=lambda s, h: run(range(1, 79)), read=read, deployed=deployed,
                         now=NOW)
    assert result[:2] == (RUN, NOW.replace(minute=0)) and old.read_bytes() == b"wet in the 00Z run"
    assert colours(tmp_path / "precip/v1/iconD2/2026111206/2026111210/12h/3/4/2.png") == {COLOUR[10]}  # 12 mm


def test_a_global_field_drops_its_180_east_column_and_wraps(tmp_path):
    model = BY_ID["icon"]._replace(zoom=1)  # 1° nodes, 361 columns: the last is 180°E again
    field = np.zeros((181, 361), np.float32)
    field[:, 0] = 1  # 180°W wet, its 180°E duplicate dry
    calls = []

    def read(source, run_time, valid, names, zoom):
        calls.append(valid)
        return Fields(-90, -180, {"precipitation": field}, {"precipitation": "mm"}, 0)

    build_model(model, tmp_path, workers=1, latest_run=lambda s, h: run(range(1, 79)), read=read, now=NOW)
    tile = tmp_path / "precip/v1/icon/2026111206/2026111210/12h/1/1/0.png"
    image = np.asarray(Image.open(tile).convert("RGBA"))
    assert tuple(image[200, 255]) == COLOUR[10]  # 12 mm at 179.65°E: the nearest node is 180°W, not the duplicate


def test_index_lists_the_manifests_in_registry_order(tmp_path):
    assert json.loads(index_json(tmp_path))["models"] == []
    for model_id in ("ecmwf", "iconD2", "ukv", "gfs"):
        (tmp_path / f"precip/v1/{model_id}").mkdir(parents=True)
        (tmp_path / f"precip/v1/{model_id}/manifest.json").write_text(json.dumps(
            {"run": "2026-11-12T06:00Z", "start": "2026-11-12T10:00Z", "windows": ["12h", "24h"], "maxZoom": 7}
        ))
    index = json.loads(index_json(tmp_path))
    assert index["legend"][0] == {"mm": 0.5, "color": "#A0D2FFBF"} and len(index["legend"]) == 11
    assert [entry["mm"] for entry in index["legend"]] == [mm for mm, _ in LEGEND]
    assert index["noData"] == "#8080804D" and list(index) == ["legend", "noData", "models"]
    assert [m["id"] for m in index["models"]] == ["ukv", "iconD2", "ecmwf"]  # registry order, no GFS
    assert index["models"][1] == {
        "id": "iconD2", "title": "ICON-D2", "km": 2.2, "bbox": [43.18, -3.94, 58.08, 20.34], "maxZoom": 7,
        "run": "2026-11-12T06:00Z", "start": "2026-11-12T10:00Z", "windows": ["12h", "24h"],
    }
    assert index["models"][0]["home"] == [49, -11, 61, 2.5]
    assert "home" not in index["models"][2]  # ECMWF: global, the default wherever no regional model applies
    assert list(index["models"][0]) == ["id", "title", "km", "bbox", "home", "maxZoom", "run", "start", "windows"]


def test_index_entry_of_the_cli(tmp_path, capsys):
    (tmp_path / "precip/v1/msm").mkdir(parents=True)
    (tmp_path / "precip/v1/msm/manifest.json").write_text(
        '{"run": "2026-11-12T00:00Z", "start": "2026-11-12T10:00Z", "windows": ["12h"], "maxZoom": 5}'
    )
    assert main(["--index", str(tmp_path)]) == 0
    assert capsys.readouterr().out == index_json(tmp_path) + "\n"


def test_main_writes_the_index_after_a_failed_model_and_leaves_other_families(tmp_path, monkeypatch, capsys):
    (tmp_path / "sky/v1").mkdir(parents=True)
    (tmp_path / "sky/v1/models.json").write_text("[]")

    def build(model, out, workers, deployed):
        if model.id == "rrfs":
            raise RuntimeError("bucket down")
        (out / f"precip/v1/{model.id}").mkdir(parents=True)
        (out / f"precip/v1/{model.id}/manifest.json").write_text(
            '{"run": "2026-11-12T06:00Z", "start": "2026-11-12T10:00Z", "windows": ["12h", "24h"], "maxZoom": 5}'
        )
        return RUN, RUN + timedelta(hours=4), ["12h", "24h"], 300, 5_000_000

    monkeypatch.setattr(build_precip, "build_model", build)
    assert main(["--out", str(tmp_path), "--models", "rrfs", "iconEu"]) == 1
    assert [m["id"] for m in json.loads((tmp_path / "precip/v1/index.json").read_text())["models"]] == ["iconEu"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["precip", "sky"]
    assert (tmp_path / "sky/v1/models.json").read_text() == "[]"
    out = capsys.readouterr().out
    assert "::error::rrfs: RuntimeError('bucket down')" in out
    assert "iconEu: run 2026-11-12T06:00Z, start 2026-11-12T10:00Z, windows 12h 24h, 300 tiles, 5 MB downloaded" in out


def test_a_deployed_run_and_start_is_skipped(tmp_path, monkeypatch, capsys):
    deployed = {"run": "2026-11-12T06:00Z", "start": "2026-11-12T10:00Z", "windows": ["12h"], "maxZoom": 3}
    read, calls = fake_read()
    latest = lambda s, h: run(range(1, 79))
    assert build_model(MODEL, tmp_path, latest_run=latest, read=read, deployed=deployed, now=NOW) is None
    assert calls == [] and not (tmp_path / "precip").exists()
    assert "iconD2: run 2026-11-12T06:00Z, start 2026-11-12T10:00Z up to date" in capsys.readouterr().out
    later = NOW + timedelta(hours=1)  # same run, the next hour: a new start
    assert build_model(MODEL, tmp_path, latest_run=latest, read=read, deployed=deployed, now=later)[1].hour == 11

    site = tmp_path / "deployed"  # the manifests pulled from the bucket
    (site / "precip/v1/ukv").mkdir(parents=True)
    (site / "precip/v1/ukv/manifest.json").write_text(json.dumps(deployed))
    got = {}

    def build(model, out, workers, deployed):
        got[model.id] = deployed

    monkeypatch.setattr(build_precip, "build_model", build)
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert main(["--out", str(tmp_path / "site"), "--models", "ukv", "rrfs", "--deployed", site.as_uri()]) == 0
    assert got == {"rrfs": None, "ukv": deployed} and output.read_text() == "changed=false\n"
