"""Aviation hazard indices on the normalized globe (row 0 at 90S, column 0 at 180W)."""

import numpy as np

from windtiles.grid import NJ, STEP_DEG

R = 6_371_000  # Earth radius, m
D = np.radians(STEP_DEG)
LAT = np.radians(-90 + STEP_DEG * np.arange(NJ))[:, None]
POLAR = np.abs(LAT[:, 0]) > np.radians(85)  # rows where x-spacing collapses

TI1_POINTS = ((2, 0.0), (4, 0.25), (8, 0.5), (12, 0.75), (16, 1.0))  # TI1 in 1e-7 s⁻² → S
CAPE_POINTS = ((100, 0.0), (1500, 1.0))  # MUCAPE J/kg → factor
RATE_POINTS = ((0.1, 0.0), (3, 1.0))  # convective rate mm/h → factor
FREEZING_K = 273.15


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


def pressure_altitude_ft(p_pa: np.ndarray) -> np.ndarray:
    """ISA pressure altitude, ft."""
    return 145366.45 * (1 - (p_pa / 101325) ** 0.190284)


def freezing_level(t2m_k: np.ndarray, sp_pa: np.ndarray, temperatures: dict[int, np.ndarray]) -> np.ndarray:
    """Lowest 0 °C crossing above ground as FL (hundreds of ft); 0 on frozen ground, NaN if none."""
    fl = np.full(np.shape(t2m_k), np.nan, np.float32)
    t0, z0 = t2m_k, pressure_altitude_ft(sp_pa)
    for hpa in sorted(temperatures, reverse=True):
        t1, z1 = temperatures[hpa], pressure_altitude_ft(hpa * 100.0)
        above = hpa * 100 < sp_pa
        cross = above & np.isnan(fl) & (t0 > FREEZING_K) & (t1 <= FREEZING_K)
        with np.errstate(divide="ignore", invalid="ignore"):  # t0 == t1 only off the crossing
            z = z0 + (z1 - z0) * (t0 - FREEZING_K) / (t0 - t1)
        fl = np.where(cross, z / 100, fl)
        t0, z0 = np.where(above, t1, t0), np.where(above, z1, z0)
    return np.where(t2m_k <= FREEZING_K, 0, fl).astype(np.float32)


def convective_rate(cp: np.ndarray, cp_before: np.ndarray) -> np.ndarray:
    """Convective precipitation rate over the last 3 h, mm/h, from accumulated cp in m."""
    return np.maximum((cp - cp_before) * 1000 / 3, 0).astype(np.float32)


def thunderstorm(mucape: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """Thunderstorm severity √(CAPE factor × rate factor)."""
    return np.sqrt(severity(mucape, CAPE_POINTS) * severity(rate, RATE_POINTS))


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
