import numpy as np
from windtiles.hazards import (
    TI1_POINTS, convective_rate, deformation, freezing_fl, icing, max_pool, severity, thunderstorm, turbulence,
)

lon, lat = np.meshgrid(np.arange(-180, 180, 0.25), np.arange(-90, 90.25, 0.25))
R = 6_371_000
x = np.radians(lon) * R * np.cos(np.radians(lat))
y = np.radians(lat) * R

def test_severity_interpolates_and_keeps_nan():
    s = severity(np.array([1, 2, 3, 4, 6, 12, 20, np.nan]), TI1_POINTS)
    assert np.allclose(s[:7], [0, 0, 0.125, 0.25, 0.375, 0.75, 1]) and np.isnan(s[7])

def test_deformation_of_stretching_and_shear():
    a = 4e-5
    assert abs(deformation(a * x, 0 * x)[360, 720] - a) < a * 0.01
    assert abs(deformation(a * y, 0 * y)[360, 720] - a) < a * 0.01

def test_deformation_wraps_across_180():
    u = 100 * np.sin(np.radians(lon))  # гладкое через 180°
    assert abs(deformation(u, 0 * u)[360, 0] - 100 / R) < 100 / R * 0.01

def test_deformation_is_nan_past_85():
    d = deformation(4e-5 * x, 0 * x)
    assert np.isnan(d[:20]).all() and np.isnan(d[701:]).all() and np.isfinite(d[20:701]).all()

def test_turbulence_of_ellrod_moderate():
    zero = 0 * x
    s = turbulence(4e-5 * x, zero, below=(zero, zero, zero + 10_000), above=(zero + 20, zero, zero + 11_000))
    assert abs(s[360, 720] - 0.5) < 0.01  # VWS 0.02 × DEF 4e-5 = 8e-7

def test_icing():
    s = icing(np.array([265.15, 265.15, 275.15, 248.15, 257.15, 271.65]), np.array([95, 82.5, 100, 100, 95, 95]))
    assert np.allclose(s, [0.7, 0.35, 0, 0, 0.35, 0.35], atol=1e-6)

def test_freezing_fl():
    fl = freezing_fl(np.array([2590.8, 2590.8, 2590.8, 0.0]), np.array([280.0, 270.0, 273.15, 280.0]))
    assert np.allclose(fl, [85.0, 0, 0, 0])

def test_convective_rate():
    assert np.allclose(convective_rate(np.array([0.001, -1e-6])), [3.6, 0])

def test_thunderstorm():
    s = thunderstorm(np.array([1500, 100, 800, 3000]), np.array([3, 5, 1.55, 0.05]))
    assert np.allclose(s, [1, 0, 0.5, 0], atol=1e-6)

def test_max_pool():
    g = np.zeros((721, 1440), np.float32); g[400, 101] = 1; g[400, 0] = 1; g[10, 500] = np.nan
    p2, p4 = max_pool(g, 2), max_pool(g, 4)
    assert p2[400, 100] == 1 and p2[400, 102] == 1 and p2[400, 103] == 0
    assert p4[400, 99] == 1 and p4[400, 104] == 0 and p4[400, 1438] == 1  # через 180°
    assert p2[10, 500] == 0 and np.isnan(max_pool(np.full((721, 1440), np.nan, np.float32), 2)[0, 0])
