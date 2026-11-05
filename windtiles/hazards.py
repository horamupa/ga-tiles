"""Aviation hazard indices on the normalized globe (row 0 at 90S, column 0 at 180W)."""

import numpy as np

from windtiles.grid import NJ, STEP_DEG

R = 6_371_000  # Earth radius, m
D = np.radians(STEP_DEG)
LAT = np.radians(-90 + STEP_DEG * np.arange(NJ))[:, None]
POLAR = np.abs(LAT[:, 0]) > np.radians(85)  # rows where x-spacing collapses

TI1_POINTS = ((2, 0.0), (4, 0.25), (8, 0.5), (12, 0.75), (16, 1.0))  # TI1 in 1e-7 s⁻² → S
CAPE_POINTS = ((100, 0.0), (1500, 1.0))  # CAPE J/kg → factor
RATE_POINTS = ((0.1, 0.0), (3, 1.0))  # convective rate mm/h → factor
FREEZING_K = 273.15
M_PER_FT = 0.3048
M_PER_SM = 1609.344
KT_PER_MS = 1.943844
CEILING_CAP_FT = 12_000  # the top of the scale: no ceiling or a higher one
VISIBILITY_CAP_SM = 10  # the top of the scale, as a METAR's 10SM
NO_CEILING_M = 19_000  # GFS writes about 20 000 gpm where it has no ceiling


def severity(values: np.ndarray, points: tuple) -> np.ndarray:
    """Piecewise-linear map through (value, S) points, clamped at both ends; NaN stays NaN."""
    xp, fp = zip(*points)
    return np.interp(values, xp, fp).astype(np.float32)


def deformation(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Total deformation √(DST² + DSH²), s⁻¹, from u, v in m/s; NaN poleward of ±85°."""
    dx = 2 * D * R * np.cos(LAT)
    dy = 2 * D * R

    def ddx(f):
        return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / dx

    def ddy(f):
        return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / dy

    d = np.hypot(ddx(u) - ddy(v), ddx(v) + ddy(u))
    d[POLAR] = np.nan
    return d.astype(np.float32)


def turbulence(u: np.ndarray, v: np.ndarray, below: tuple, above: tuple) -> np.ndarray:
    """Ellrod TI1 severity; below/above are (u, v, gh) of the neighbouring levels."""
    (ub, vb, zb), (ua, va, za) = below, above
    vws = np.hypot(ua - ub, va - vb) / (za - zb)
    return severity(vws * deformation(u, v) * 1e7, TI1_POINTS)


def icing(t_k: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Icing potential 0.7 × f(T) × g(RH), T in K, RH in %."""
    f = np.interp(t_k - FREEZING_K, (-20, -12, -3, 0), (0, 1, 1, 0))
    g = np.interp(rh, (70, 95), (0, 1))
    return (0.7 * f * g).astype(np.float32)


def freezing_fl(height_m: np.ndarray, t2m_k: np.ndarray) -> np.ndarray:
    """0 °C isotherm in hundreds of feet above sea level; 0 on frozen ground and where GFS gives no isotherm."""
    return np.where(t2m_k <= FREEZING_K, 0, height_m / M_PER_FT / 100).astype(np.float32)


def ceiling_ft(ceiling_m: np.ndarray, orography_m: np.ndarray) -> np.ndarray:
    """Ceiling above ground, ft, 0…`CEILING_CAP_FT`. GFS gives it above sea level: it never lies under its own terrain."""
    above_ground = (ceiling_m - orography_m) / M_PER_FT
    return np.clip(np.where(ceiling_m >= NO_CEILING_M, CEILING_CAP_FT, above_ground), 0, CEILING_CAP_FT).astype(np.float32)


def visibility_sm(visibility_m: np.ndarray) -> np.ndarray:
    """Surface visibility, statute miles, 0…`VISIBILITY_CAP_SM`."""
    return np.clip(visibility_m / M_PER_SM, 0, VISIBILITY_CAP_SM).astype(np.float32)


def gust_kt(gust_ms: np.ndarray) -> np.ndarray:
    """Surface gust, kt."""
    return (gust_ms * KT_PER_MS).astype(np.float32)


def min_pool(globe: np.ndarray, stride: int) -> np.ndarray:
    """`max_pool` of the lowest: a fog pocket between coarse nodes must not read as clear."""
    return -max_pool(-globe, stride)


def convective_rate(cprat: np.ndarray) -> np.ndarray:
    """Convective precipitation rate, mm/h, from kg/m²/s."""
    return np.maximum(cprat * 3600, 0).astype(np.float32)


def thunderstorm(cape: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """Thunderstorm severity √(CAPE factor × rate factor)."""
    return np.sqrt(severity(cape, CAPE_POINTS) * severity(rate, RATE_POINTS))


def max_pool(globe: np.ndarray, stride: int) -> np.ndarray:
    """Max over a ±stride // 2 node window, wrapping in longitude, clipped in latitude; NaN ignored."""
    h = stride // 2
    rows = globe
    for s in range(1, h + 1):
        rows = np.fmax(rows, np.fmax(np.roll(globe, s, axis=1), np.roll(globe, -s, axis=1)))
    padded = np.pad(rows, ((h, h), (0, 0)), constant_values=np.nan)
    out = rows
    for s in range(-h, h + 1):
        out = np.fmax(out, padded[h + s : h + s + len(rows)])
    return out
