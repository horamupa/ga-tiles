from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
import eccodes
import numpy as np
import pytest
from windtiles import source
from windtiles.source import IdxEntry, byte_ranges, file_url, latest_run, parse_idx, read_fields, wanted

RUN = datetime(2026, 11, 5, 12, tzinfo=timezone.utc)
ZERO = {"shortName": "gh", "typeOfLevel": "isothermZero"}
CAPE = {  # NCEP table: the keys of the real GFS message, `shortName` alone does not reach it
    "centre": "kwbc", "parameterCategory": 7, "parameterNumber": 6,
    "typeOfFirstFixedSurface": 108, "scaledValueOfFirstFixedSurface": 18000,
    "typeOfSecondFixedSurface": 108, "scaleFactorOfSecondFixedSurface": 0, "scaledValueOfSecondFixedSurface": 0,
}
CPRAT = {"shortName": "cpr", "typeOfLevel": "surface"}
HPBL = {"centre": "kwbc", "discipline": 0, "parameterCategory": 3, "parameterNumber": 196, "typeOfFirstFixedSurface": 1}

def grib(path, ni=1440, nj=721, step_deg=0.25, first_lat=90.0, keys=None, values=None):
    handle = eccodes.codes_grib_new_from_samples("regular_ll_pl_grib2")
    eccodes.codes_set_key_vals(handle, {
        "shortName": "u", "level": 250, "step": 6, "Ni": ni, "Nj": nj,
        "iDirectionIncrementInDegrees": step_deg, "jDirectionIncrementInDegrees": step_deg,
        "latitudeOfFirstGridPointInDegrees": first_lat,
        "latitudeOfLastGridPointInDegrees": first_lat - (nj - 1) * step_deg,
        "longitudeOfFirstGridPointInDegrees": 180.0,
        "longitudeOfLastGridPointInDegrees": 180.0 + (ni - 1) * step_deg - 360,
        **(keys or {}),
    })
    rows = np.repeat(np.arange(nj, dtype=np.float64)[:, None] / 10, ni, axis=1)  # value = GRIB row / 10
    eccodes.codes_set_values(handle, rows.ravel() if values is None else values)
    with open(path, "wb") as file:
        eccodes.codes_write(handle, file)
    eccodes.codes_release(handle)
    return path

def entry(name, level, forecast="6 hour fcst", start=0, end=None):
    return IdxEntry(name, level, forecast, start, end)

def test_exists_is_a_head_request():
    class Handler(BaseHTTPRequestHandler):  # only HEAD: a GET gets 501 and reads as absent
        def do_HEAD(self):
            self.send_response(200 if self.path == "/f066.idx" else 404)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert source.exists(f"{base}/f066.idx") is True
        assert source.exists(f"{base}/f069.idx") is False
    finally:
        server.shutdown()
        server.server_close()
    assert source.exists(f"{base}/f066.idx") is False  # nothing listens now

def test_file_url():
    assert file_url(RUN, 66) == (
        "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20261105/12/atmos/gfs.t12z.pgrb2.0p25.f066"
    )

@pytest.mark.parametrize("ready, expected", [
    ([12], datetime(2026, 11, 5, 12, tzinfo=timezone.utc)),
    ([6], datetime(2026, 11, 5, 6, tzinfo=timezone.utc)),
    ([], None),
])
def test_latest_run_takes_the_newest_with_step_66(ready, expected):
    now = datetime(2026, 11, 5, 16, 12, tzinfo=timezone.utc)
    urls = {file_url(now.replace(hour=hour, minute=0), 66) + ".idx" for hour in ready}
    assert latest_run(now, urls.__contains__) == expected

def test_latest_run_looks_back_24_hours():
    now = datetime(2026, 11, 5, 16, 12, tzinfo=timezone.utc)
    seen = []
    latest_run(now, lambda url: seen.append(url) or False)
    assert seen[0].endswith("gfs.20261105/12/atmos/gfs.t12z.pgrb2.0p25.f066.idx")
    assert seen[-1].endswith("gfs.20261104/12/atmos/gfs.t12z.pgrb2.0p25.f066.idx")
    assert len(seen) == 5

def test_parse_idx():
    text = (
        "1:0:d=2026110406:HGT:1 hybrid level:6 hour fcst:\n"
        "2:1035461:d=2026110406:TMP:2 m above ground:6 hour fcst:\n"
        "3:2100000:d=2026110406:CPRAT:surface:0-6 hour ave fcst:\n"
    )
    assert parse_idx(text) == [
        IdxEntry("HGT", "1 hybrid level", "6 hour fcst", 0, 1035460),
        IdxEntry("TMP", "2 m above ground", "6 hour fcst", 1035461, 2099999),
        IdxEntry("CPRAT", "surface", "0-6 hour ave fcst", 2100000, None),
    ]

@pytest.mark.parametrize("name, level, forecast, expected", [
    ("CPRAT", "surface", "6 hour fcst", True),
    ("CPRAT", "surface", "0-6 hour ave fcst", False),
    ("UGRD", "850 mb", "6 hour fcst", True),
    ("UGRD", "875 mb", "6 hour fcst", False),
    ("TMP", "2 m above ground", "6 hour fcst", True),
    ("TMP", "2 mb", "6 hour fcst", False),
    ("UGRD", "10 mb", "6 hour fcst", False),
    ("UGRD", "850 mb", "9 hour fcst", False),
])
def test_wanted_takes_the_instant_rate(name, level, forecast, expected):
    assert wanted(entry(name, level, forecast), 6) is expected

@pytest.mark.parametrize("name, level, expected", [
    ("UGRD", "250 mb", True), ("TMP", "250 mb", False), ("RH", "600 mb", True), ("RH", "500 mb", False),
    ("HGT", "250 mb", True), ("HGT", "700 mb", True), ("HGT", "50 mb", False), ("HGT", "0C isotherm", True),
    ("TCDC", "100 mb", True), ("TCDC", "50 mb", False), ("TCDC", "entire atmosphere", True), ("HPBL", "surface", True),
    ("CAPE", "180-0 mb above ground", True), ("CAPE", "surface", False), ("PRES", "surface", True),
])
def test_wanted_follows_the_field_table(name, level, expected):
    assert wanted(entry(name, level), 6) is expected

def test_byte_ranges_join_neighbours():
    entries = [
        entry("A", "1", start=0, end=99), entry("B", "1", start=100, end=199),
        entry("C", "1", start=300, end=399), entry("D", "1", start=400, end=None),
    ]
    assert byte_ranges(entries) == [(0, 199), (300, None)]

def idx_of(rows):
    return "".join(
        f"{number}:{(number - 1) * 100}:d=2026110512:{name}:{level}:{forecast}:\n"
        for number, (name, level, forecast) in enumerate(rows, start=1)
    )

def test_download_asks_only_wanted_ranges(tmp_path):
    first = [("UGRD", "850 mb"), ("VGRD", "850 mb")]
    last = ("CPRAT", "surface")
    idx = idx_of([
        ("HGT", "1 hybrid level", "6 hour fcst"),
        *((name, level, "6 hour fcst") for name, level in first),
        ("CPRAT", "surface", "0-6 hour ave fcst"),
        (*last, "6 hour fcst"),
        *((name, level, "6 hour fcst") for name, level in sorted(source.FIELDS - {*first, last})),
    ])
    calls = []

    def fetch(url, range_header):
        calls.append((url, range_header))
        return idx.encode() if url.endswith(".idx") else range_header.encode()

    paths = source.download(RUN, [6], tmp_path / "work", fetch)
    url = file_url(RUN, 6)
    assert calls == [(url + ".idx", None), (url, "bytes=100-299"), (url, "bytes=400-")]
    assert paths == [tmp_path / "work" / "f006.grib2"]
    assert paths[0].read_bytes() == b"bytes=100-299bytes=400-"

def test_download_refuses_a_step_without_a_wanted_field(tmp_path):
    complete = {(name, level, "9 hour fcst") for name, level in source.FIELDS}
    idx = idx_of(sorted(complete - {("CPRAT", "surface", "9 hour fcst")}))

    def fetch(url, range_header):
        assert url.endswith(".idx")  # nothing is fetched for a step that cannot be complete
        return idx.encode()

    with pytest.raises(ValueError, match=r"step 9: .*CPRAT"):
        source.download(RUN, [9], tmp_path / "work", fetch)

def test_download_goes_on_without_a_sky_field(tmp_path, capsys):
    idx = idx_of(sorted((name, level, "6 hour fcst") for name, level in source.FIELDS))
    source.download(RUN, [6], tmp_path / "work", lambda url, range_header: idx.encode())
    assert "step 6: no [('HCDC', 'high cloud layer')" in capsys.readouterr().out

def test_reads_and_normalizes_a_global_field(tmp_path):
    fields = read_fields([grib(tmp_path / "pl.grib2")])
    globe = fields[("u", 250, 6)]
    assert globe.shape == (721, 1440)
    assert globe[0, 0] == pytest.approx(72, abs=0.1)   # south pole is the last GRIB row
    assert globe[720, 0] == pytest.approx(0, abs=0.1)

@pytest.mark.parametrize("keys, valid, key", [
    ({}, 10.0, ("u", 250, 6)),
    ({"shortName": "gh", "level": 500}, 5_000.0, ("gh", 500, 6)),
    ({"shortName": "10u", "typeOfLevel": "heightAboveGround", "level": 10}, 10.0, ("10u", 0, 6)),
    ({"shortName": "2t", "typeOfLevel": "heightAboveGround", "level": 2}, 280.0, ("2t", 0, 6)),
    ({"shortName": "sp", "typeOfLevel": "surface"}, 100_000.0, ("sp", 0, 6)),
    (ZERO, 3_000.0, ("zero", 0, 6)),
    (CAPE, 500.0, ("cape", 0, 6)),
    (CPRAT, 0.001, ("cprat", 0, 6)),
    (HPBL, 800.0, ("hpbl", 0, 6)),
])
def test_names_the_fields(tmp_path, keys, valid, key):
    path = grib(tmp_path / "field.grib2", keys=keys, values=np.full(1440 * 721, valid))
    assert set(read_fields([path])) == {key}

@pytest.mark.parametrize("grid", [{"ni": 720, "step_deg": 0.5}, {"nj": 361, "step_deg": 0.5}, {"first_lat": 89.75}])
def test_rejects_a_foreign_grid(tmp_path, grid):
    with pytest.raises(ValueError, match="u 850 6"):
        read_fields([grib(tmp_path / "bad.grib2", keys={"level": 850}, **grid)])

@pytest.mark.parametrize("keys", [{"iScansNegatively": 1}, {"jScansPositively": 1}])
def test_rejects_a_flipped_scan(tmp_path, keys):
    with pytest.raises(ValueError, match="u 850 6: scanned"):
        read_fields([grib(tmp_path / "flip.grib2", keys={"level": 850, **keys})])

@pytest.mark.parametrize("keys, valid, value", [
    ({"level": 850}, 10.0, 150.0), ({"level": 850}, 10.0, -151.0), ({"shortName": "t", "level": 850}, 250.0, 140.0),
])
def test_rejects_values_out_of_range(tmp_path, keys, valid, value):
    values = np.full(1440 * 721, valid)
    values[100] = value
    with pytest.raises(ValueError, match="out of range"):
        read_fields([grib(tmp_path / "range.grib2", keys=keys, values=values)])

@pytest.mark.parametrize("keys, valid, value, name", [
    ({"shortName": "gh"}, 10_000.0, 30_000, "gh 250 6"), ({"shortName": "r"}, 50.0, 200, "r 250 6"),
    ({"shortName": "r"}, 50.0, -11, "r 250 6"),
    (ZERO, 3_000.0, 11_000, "zero 0 6"), (ZERO, 3_000.0, -1_100, "zero 0 6"),
    (CAPE, 500.0, 25_000, "cape 0 6"), (CAPE, 500.0, -2, "cape 0 6"),
    (CPRAT, 0.001, 0.2, "cprat 0 6"), (CPRAT, 0.001, -1e-5, "cprat 0 6"),
])
def test_skips_a_hazard_value_out_of_range(tmp_path, capsys, keys, valid, value, name):
    values = np.full(1440 * 721, valid)
    values[100] = value
    fields = read_fields([grib(tmp_path / "pl.grib2"), grib(tmp_path / "range.grib2", keys=keys, values=values)])
    assert set(fields) == {("u", 250, 6)}  # the wind field next to it is still read
    assert capsys.readouterr().out == f"skip {name}: values out of range\n"

def with_gap(valid, gap=9999.0):
    values = np.full(1440 * 721, valid)
    values[100] = gap
    return values

@pytest.mark.parametrize("keys, valid, key", [
    (ZERO, 3_000.0, ("zero", 0, 6)), (CAPE, 500.0, ("cape", 0, 6)), (CPRAT, 0.001, ("cprat", 0, 6)),
    ({"shortName": "gh", "level": 500}, 5_000.0, ("gh", 500, 6)),
    ({"shortName": "r", "level": 700}, 50.0, ("r", 700, 6)),
])
def test_missing_values_are_nan_in_a_hazard_field(tmp_path, keys, valid, key):
    gap = {"bitmapPresent": 1, "missingValue": 9999}
    path = grib(tmp_path / "gap.grib2", keys={**keys, **gap}, values=with_gap(valid))
    globe = read_fields([path])[key]
    assert np.isnan(globe).sum() == 1
    assert np.nanmax(globe) == pytest.approx(valid)

def test_a_gap_does_not_hide_a_hazard_value_out_of_range(tmp_path, capsys):
    values = with_gap(500.0)
    values[200] = 25_000
    path = grib(tmp_path / "gap.grib2", keys={**CAPE, "bitmapPresent": 1, "missingValue": 9999}, values=values)
    assert read_fields([path]) == {}
    assert capsys.readouterr().out == "skip cape 0 6: values out of range\n"

@pytest.mark.parametrize("keys, valid", [
    ({"level": 850}, 10.0), ({"shortName": "t", "level": 850}, 250.0),
    ({"shortName": "2t", "typeOfLevel": "heightAboveGround", "level": 2}, 280.0),
    ({"shortName": "sp", "typeOfLevel": "surface"}, 100_000.0),
])
def test_missing_values_stop_a_wind_field(tmp_path, keys, valid):
    gap = {"bitmapPresent": 1, "missingValue": 9999}
    with pytest.raises(ValueError, match="has missing"):
        read_fields([grib(tmp_path / "gap.grib2", keys={**keys, **gap}, values=with_gap(valid))])
