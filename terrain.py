"""Procedural sand dune terrain generation via sum of sinusoids."""

import numpy as np


class Terrain:
    """1D height field y = offset + base_slope*x + sum(A_i * sin(k_i * x + phi_i))."""

    def __init__(self, name, amplitudes, wavenumbers, phases, offset=0.0, base_slope=0.0):
        self.name = name
        self.amplitudes = np.array(amplitudes, dtype=np.float64)
        self.wavenumbers = np.array(wavenumbers, dtype=np.float64)
        self.phases = np.array(phases, dtype=np.float64)
        self.offset = offset
        self.base_slope = base_slope

    def __call__(self, x):
        """Return terrain height at x (scalar or array)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.full_like(x, self.offset + self.base_slope * x, dtype=np.float64)
        for A, k, phi in zip(self.amplitudes, self.wavenumbers, self.phases):
            y += A * np.sin(k * x + phi)
        return y

    def slope(self, x):
        """Return f'(x) (scalar or array)."""
        x = np.asarray(x, dtype=np.float64)
        dy = np.full_like(x, self.base_slope, dtype=np.float64)
        for A, k, phi in zip(self.amplitudes, self.wavenumbers, self.phases):
            dy += A * k * np.cos(k * x + phi)
        return dy

    def normal(self, x):
        """Return outward (upward-pointing) normal vector at surface point (x, f(x))."""
        s = self.slope(x)
        # tangent = (1, s), normal perpendicular to tangent pointing up: (-s, 1)
        norm = np.sqrt(s * s + 1.0)
        return -s / norm, 1.0 / norm

    def __repr__(self):
        return f"Terrain({self.name!r})"


# ---------------------------------------------------------------------------
# Preset terrains
# ---------------------------------------------------------------------------

PRESETS = {
    "flat": Terrain("flat", [], [], [], offset=0.0),

    "gentle_dunes": Terrain(
        "gentle_dunes",
        amplitudes=[12.0, 8.0, 5.0],
        wavenumbers=[0.02, 0.04, 0.07],
        phases=[0.0, 1.0, 2.5],
        offset=0.0,
    ),

    "steep_dunes": Terrain(
        "steep_dunes",
        amplitudes=[25.0, 18.0, 12.0, 8.0],
        wavenumbers=[0.015, 0.03, 0.05, 0.08],
        phases=[0.5, 2.0, 0.0, 3.0],
        offset=0.0,
    ),

    "uphill": Terrain(
        "uphill",
        amplitudes=[15.0, 8.0],
        wavenumbers=[0.012, 0.035],
        phases=[0.0, 1.5],
        offset=0.0,
        base_slope=0.08,  # steady climb
    ),

    "valley": Terrain(
        "valley",
        amplitudes=[22.0, 10.0, 6.0],
        wavenumbers=[0.018, 0.04, 0.08],
        phases=[np.pi, 0.0, 1.0],
        offset=2.0,
    ),

    "links_course": Terrain(
        "links_course",
        amplitudes=[14.0, 11.0, 8.0, 5.0, 3.0],
        wavenumbers=[0.02, 0.038, 0.06, 0.085, 0.11],
        phases=[0.3, 1.7, 3.1, 4.5, 5.9],
        offset=0.0,
    ),
}


def get_preset(name):
    """Return a terrain by preset name."""
    if name not in PRESETS:
        raise KeyError(f"Unknown terrain preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]
