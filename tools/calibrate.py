"""Check hazard thresholds of the latest ECMWF run against AWC PIREPs/AIREPs and convective SIGMETs.

Not part of the package or CI. Run from the repository root with an isolated interpreter:

    .venv/bin/python -I tools/calibrate.py --work "$TMPDIR/wx-calib"

`--work` holds the GRIB downloads and AWC answers of one run (outside the repository); a rerun with the
same directory reuses them. Prints, for the current thresholds and for thresholds × 0.75, 1.5 and 2:
MOD+ turbulence reports at FL200+ inside S ≥ 0.25, the share of the globe with turbulence S ≥ 0.5 on
300/250/200 hPa, and the share of convective SIGMET area under thunderstorm S ≥ 0.25.
"""

import argparse
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from windtiles import build, hazards, source
from windtiles.grid import NI, NJ, STEP_DEG

STEPS = [6, 9, 12, 15]
FACTORS = [1, 0.75, 1.5, 2]
AWC = "https://aviationweather.gov/api/data/"
PIREP_URL = AWC + "pirep?format=json&bbox=-90,-180,90,180&age=3&inten=lgt"  # HazardService.pilotReportRequest
SIGMET_URL = AWC + "sigmet?format=json"  # HazardService.sigmetRequest(hour: 0)
REPORT_LEVELS = [400, 300, 250, 200, 150]  # hPa, nearest to the report's FL
AREA_LEVELS = [300, 250, 200]
MODERATE = {"MOD", "MOD-SEV", "SEV", "SEV-EXTM", "EXTM"}
LIGHT_MODERATE = MODERATE | {"LGT-MOD"}
CONVECTIVE = {"TS", "CONV", "CB"}
WINDOW = round(0.5 / STEP_DEG)  # ±0.5° in nodes
WEIGHT = np.broadcast_to(np.cos(hazards.LAT), (NJ, NI))
LAT_DEG = -90 + STEP_DEG * np.arange(NJ)
LON_DEG = -180 + STEP_DEG * np.arange(NI)


def fetch_json(url: str, path: Path):
    """AWC answer, cached in `path`; 204 or an empty body is an empty list."""
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "wind-tiles calibrate"})
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read() or b"[]")
    return json.loads(path.read_text() or "[]")


def pressure_hpa(fl: float) -> float:
    """ISA pressure of a flight level (hundreds of ft)."""
    return 1013.25 * (1 - fl * 100 / 145366.45) ** (1 / 0.190284)


def turbulence_reports(pireps: list[dict], intensities: set[str]) -> list[dict]:
    """Reports at FL200+ with turbulence of one of `intensities` in either group."""
    return [
        p for p in pireps
        if (p.get("fltLvl") or 0) >= 200 and p.get("lat") is not None and p.get("lon") is not None
        and {p.get("tbInt1"), p.get("tbInt2")} & intensities
    ]


def convective_rings(sigmets: list[dict]) -> list[list[tuple[float, float]]]:
    """(lon, lat) rings of convective SIGMETs, each distinct ring once; rings under 3 points dropped."""
    rings = []
    for s in sigmets:
        if s.get("hazard") not in CONVECTIVE and "CB" not in (s.get("qualifier") or ""):
            continue
        coords = s.get("coords") or []
        for ring in coords if coords and isinstance(coords[0], list) else [coords]:
            points = [(float(c["lon"]), float(c["lat"])) for c in ring]
            if len(set(points)) >= 3 and points not in rings:
                rings.append(points)
    return rings


def ring_mask(ring: list[tuple[float, float]]) -> np.ndarray:
    """Globe nodes inside a ring (even-odd ray casting); a ring crossing 180° is unwrapped east."""
    lon, lat = np.array(ring).T
    if lon.max() - lon.min() > 180:
        lon = np.where(lon < 0, lon + 360, lon)
    rows = np.flatnonzero((LAT_DEG >= lat.min()) & (LAT_DEG <= lat.max()))
    mask = np.zeros((NJ, NI), bool)
    for shift in (0, 360):
        columns = np.flatnonzero((LON_DEG + shift >= lon.min()) & (LON_DEG + shift <= lon.max()))
        if not len(rows) or not len(columns):
            continue
        x, y = np.meshgrid(LON_DEG[columns] + shift, LAT_DEG[rows])
        inside = np.zeros(x.shape, bool)
        for (x0, y0), (x1, y1) in zip(zip(lon, lat), zip(np.roll(lon, -1), np.roll(lat, -1))):
            crosses = (y0 > y) != (y1 > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            inside ^= crosses & (x < x_cross)
        mask[np.ix_(rows, columns)] |= inside
    return mask


def scaled(points: tuple, factor: float) -> tuple:
    return tuple((x * factor, s) for x, s in points)


def severities(fields, factor: float) -> tuple[dict, dict]:
    """Turbulence S per (hPa, step) and thunderstorm S per step with every threshold × factor."""
    base = hazards.TI1_POINTS, hazards.CAPE_POINTS, hazards.RATE_POINTS
    hazards.TI1_POINTS, hazards.CAPE_POINTS, hazards.RATE_POINTS = (scaled(p, factor) for p in base)
    try:
        turbulence = {(hpa, step): build.hazard_fields(fields, str(hpa), step)[0] for hpa in REPORT_LEVELS for step in STEPS}
        storms = {step: build.hazard_fields(fields, "col", step)[0] for step in STEPS}
    finally:
        hazards.TI1_POINTS, hazards.CAPE_POINTS, hazards.RATE_POINTS = base
    return turbulence, storms


def hits(reports: list[dict], turbulence: dict, run: datetime) -> int:
    """Reports with S ≥ 0.25 within ±0.5° and ±1 level of the nearest frame and level."""
    count = 0
    for p in reports:
        step = min(STEPS, key=lambda s: abs(p["obsTime"] - (run + timedelta(hours=s)).timestamp()))
        index = min(range(len(REPORT_LEVELS)), key=lambda i: abs(REPORT_LEVELS[i] - pressure_hpa(p["fltLvl"])))
        row = round((p["lat"] + 90) / STEP_DEG)
        rows = slice(max(row - WINDOW, 0), row + WINDOW + 1)
        columns = (round((p["lon"] + 180) / STEP_DEG) + np.arange(-WINDOW, WINDOW + 1)) % NI
        levels = REPORT_LEVELS[max(index - 1, 0) : index + 2]
        count += any(np.nanmax(turbulence[(hpa, step)][rows][:, columns], initial=0) >= 0.25 for hpa in levels)
    return count


def area_share(globe: np.ndarray, threshold: float, mask=None) -> float:
    """cos φ-weighted share of the globe (or of `mask`) with S ≥ threshold; NaN counts as below."""
    weight = WEIGHT if mask is None else WEIGHT * mask
    return float((weight * (globe >= threshold)).sum() / weight.sum())


def nearest_step(run: datetime, time: datetime) -> int:
    return min(STEPS, key=lambda s: abs(run + timedelta(hours=s) - time))


def main(argv=None) -> int:
    from ecmwf.opendata import Client

    parser = argparse.ArgumentParser(prog="calibrate")
    parser.add_argument("--work", type=Path, required=True, help="download directory, outside the repository")
    args = parser.parse_args(argv)

    client = Client(source=source.SOURCE)
    run = source.latest_run(client)
    work = args.work / f"{run:%Y%m%d%H}"
    paths = [work / "pl.grib2", work / "sfc.grib2"]
    if not all(p.exists() for p in paths):
        paths = source.download(client, run, STEPS, work)
    fields = source.read_fields(paths)
    pireps = fetch_json(PIREP_URL, work / "pirep.json")
    sigmets = fetch_json(SIGMET_URL, work / "sigmet.json")
    asked = datetime.fromtimestamp((work / "sigmet.json").stat().st_mtime, timezone.utc)

    moderate = turbulence_reports(pireps, MODERATE)
    light_moderate = turbulence_reports(pireps, LIGHT_MODERATE)
    rings = convective_rings(sigmets)
    storm_mask = np.zeros((NJ, NI), bool)
    for ring in rings:
        storm_mask |= ring_mask(ring)
    storm_step = nearest_step(run, asked)

    print(f"run {run:%Y-%m-%dT%HZ}, steps {STEPS}, AWC asked {asked:%Y-%m-%d %H:%MZ}")
    print(f"PIREP/AIREP: {len(pireps)} total, FL200+ MOD+ {len(moderate)}, FL200+ LGT-MOD+ {len(light_moderate)}")
    print(f"convective SIGMET rings: {len(rings)}, {storm_mask.sum()} nodes, frame +{storm_step} h")
    print()
    print(f"{'factor':>6} {'MOD+ hits':>12} {'LGT-MOD+ hits':>14} {'S≥0.5 area 300/250/200':>24} {'SIGMET under S≥0.25':>20}")
    for factor in FACTORS:
        turbulence, storms = severities(fields, factor)
        areas = [np.mean([area_share(turbulence[(hpa, s)], 0.5) for s in STEPS]) for hpa in AREA_LEVELS]
        h = hits(moderate, turbulence, run)
        h_lm = hits(light_moderate, turbulence, run)
        cover = area_share(storms[storm_step], 0.25, storm_mask) if storm_mask.any() else float("nan")
        print(
            f"{factor:>6} {h:>4}/{len(moderate):<3} {h / max(len(moderate), 1):>4.0%}"
            f" {h_lm:>4}/{len(light_moderate):<3} {h_lm / max(len(light_moderate), 1):>4.0%}"
            f" {' / '.join(f'{a:.1%}' for a in areas):>24} {cover:>20.0%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
