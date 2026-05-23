"""Parameter sweep over (v0, theta) grid for a given terrain."""

import numpy as np
from simulation import simulate_shot


class SweepResult:
    """Container for parameter sweep output."""

    def __init__(self, terrain_name, v0_grid, theta_grid,
                 landing_x, final_x, max_height, num_bounces, flight_time=None):
        self.terrain_name = terrain_name
        self.v0_grid = v0_grid        # 1D array of v0 values
        self.theta_grid = theta_grid  # 1D array of theta values (degrees)
        self.landing_x = landing_x    # 2D array [v0_idx, theta_idx]
        self.final_x = final_x        # 2D array
        self.max_height = max_height  # 2D array
        self.num_bounces = num_bounces  # 2D array

    @property
    def shape(self):
        return self.landing_x.shape

    @property
    def carry_distance(self):
        """Distance from tee to first impact."""
        return self.landing_x

    @property
    def total_distance(self):
        """Distance from tee to final resting position."""
        return self.final_x

    @property
    def roll_distance(self):
        """Distance rolled after first impact."""
        return self.final_x - self.landing_x


def parameter_sweep(terrain, v0_range=(20, 70), theta_range=(5, 60),
                    resolution=50, tee_height=0.0, dt=None,
                    include_roll=True, verbose=True):
    """Run a full sweep over the (v0, theta) parameter grid.

    Args:
        terrain: Terrain instance
        v0_range: (min, max) initial speed in m/s
        theta_range: (min, max) launch angle in degrees
        resolution: number of points per axis (N x N grid)
        tee_height: tee height above terrain at x=0
        dt: RK4 time step
        include_roll: whether to simulate rolling
        verbose: print progress

    Returns:
        SweepResult object
    """
    v0_vals = np.linspace(v0_range[0], v0_range[1], resolution)
    theta_vals = np.linspace(theta_range[0], theta_range[1], resolution)

    n_v0 = len(v0_vals)
    n_th = len(theta_vals)
    total = n_v0 * n_th

    landing_x = np.full((n_v0, n_th), np.nan)
    final_x = np.full((n_v0, n_th), np.nan)
    max_height = np.full((n_v0, n_th), np.nan)
    num_bounces = np.full((n_v0, n_th), np.nan)

    count = 0
    for i, v0 in enumerate(v0_vals):
        for j, theta in enumerate(theta_vals):
            result = simulate_shot(
                v0, theta, terrain,
                tee_height=tee_height, dt=dt, include_roll=include_roll
            )
            landing_x[i, j] = result['landing_x']
            final_x[i, j] = result['final_x']
            max_height[i, j] = result['max_height']
            num_bounces[i, j] = result['num_bounces']

            count += 1
            if verbose and count % max(1, total // 20) == 0:
                pct = 100.0 * count / total
                print(f"  [{terrain.name}] {count}/{total} shots ({pct:.0f}%)")

    if verbose:
        print(f"  [{terrain.name}] complete: {total} shots simulated")

    return SweepResult(
        terrain_name=terrain.name,
        v0_grid=v0_vals,
        theta_grid=theta_vals,
        landing_x=landing_x,
        final_x=final_x,
        max_height=max_height,
        num_bounces=num_bounces,
    )
