"""Visualization functions for golf simulation results."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colormaps


# Consistent style
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
})


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Plot 1: Landing position heatmap per terrain
# ---------------------------------------------------------------------------

def plot_landing_heatmap(sweep_result, output_dir):
    """Heatmap: (theta, v0) -> final_x with contour lines."""
    sr = sweep_result
    fig, ax = plt.subplots(figsize=(8, 6))

    X, Y = np.meshgrid(sr.theta_grid, sr.v0_grid)
    Z = sr.final_x

    # Mask invalid (NaN) regions
    Z_masked = np.ma.masked_invalid(Z)

    im = ax.pcolormesh(X, Y, Z_masked, cmap='viridis', shading='auto')

    # Contour lines at round distances
    levels = np.arange(0, 401, 25)
    cs = ax.contour(X, Y, Z_masked, levels=levels, colors='white',
                    linewidths=0.5, alpha=0.6)
    ax.clabel(cs, cs.levels[::2], inline=True, fontsize=7, fmt='%.0f m')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Final Position X (m)')

    ax.set_xlabel('Launch Angle θ (deg)')
    ax.set_ylabel('Initial Speed v₀ (m/s)')
    ax.set_title(f'Shot Parameter → Landing Position\nTerrain: {sr.terrain_name}')

    fig.tight_layout()
    path = os.path.join(output_dir, f'heatmap_{sr.terrain_name}.png')
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 2: Contour comparison across terrains
# ---------------------------------------------------------------------------

def plot_contour_comparison(sweep_results, output_dir):
    """Overlay contour lines from multiple terrains on same axes."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    levels = np.arange(0, 401, 25)

    for sr, color in zip(sweep_results, colors):
        X, Y = np.meshgrid(sr.theta_grid, sr.v0_grid)
        Z = np.ma.masked_invalid(sr.final_x)
        cs = ax.contour(X, Y, Z, levels=levels, colors=color, linewidths=1.2)
        # Label one contour per terrain
        mid_level = levels[len(levels) // 2]
        try:
            ax.clabel(cs, [mid_level], inline=True, fontsize=8, fmt='%.0f m')
        except (ValueError, IndexError):
            pass

    # Custom legend
    from matplotlib.lines import Line2D
    legend_lines = [Line2D([0], [0], color=c, lw=2, label=sr.terrain_name)
                    for sr, c in zip(sweep_results, colors)]
    ax.legend(handles=legend_lines, loc='upper right', fontsize=8)

    ax.set_xlabel('Launch Angle θ (deg)')
    ax.set_ylabel('Initial Speed v₀ (m/s)')
    ax.set_title('Landing Position Contours: Terrain Comparison')
    ax.set_xlim(sweep_results[0].theta_grid[0], sweep_results[0].theta_grid[-1])
    ax.set_ylim(sweep_results[0].v0_grid[0], sweep_results[0].v0_grid[-1])

    fig.tight_layout()
    path = os.path.join(output_dir, 'contour_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 3: Difference map between two terrains
# ---------------------------------------------------------------------------

def plot_difference_map(sr_a, sr_b, output_dir):
    """Heatmap of delta = landing_x[A] - landing_x[B]."""
    fig, ax = plt.subplots(figsize=(8, 6))

    X, Y = np.meshgrid(sr_a.theta_grid, sr_a.v0_grid)
    delta = sr_a.final_x - sr_b.final_x

    vmax = max(abs(np.nanmin(delta)), abs(np.nanmax(delta)))
    im = ax.pcolormesh(X, Y, delta, cmap='RdBu_r', shading='auto',
                        vmin=-vmax, vmax=vmax)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f'Δ Final X (m)\n({sr_a.terrain_name} − {sr_b.terrain_name})')

    ax.set_xlabel('Launch Angle θ (deg)')
    ax.set_ylabel('Initial Speed v₀ (m/s)')
    ax.set_title(f'Terrain Difference Map\n{sr_a.terrain_name} vs {sr_b.terrain_name}')

    fig.tight_layout()
    path = os.path.join(output_dir, f'diff_{sr_a.terrain_name}_vs_{sr_b.terrain_name}.png')
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 4: Cross-section — fixed launch angles, distance vs speed
# ---------------------------------------------------------------------------

def plot_cross_section_angle(sweep_results, fixed_thetas=None, output_dir=None):
    """landing_x vs v0 for selected fixed launch angles. One line per terrain per angle."""
    if fixed_thetas is None:
        fixed_thetas = [10, 20, 30, 40, 50]

    n_angles = len(fixed_thetas)
    fig, axes = plt.subplots(1, n_angles, figsize=(3.5 * n_angles, 4),
                             squeeze=False)

    for ax_idx, theta_target in enumerate(fixed_thetas):
        ax = axes[0, ax_idx]
        for sr in sweep_results:
            # Find closest theta index
            th_idx = np.argmin(np.abs(sr.theta_grid - theta_target))
            ax.plot(sr.v0_grid, sr.final_x[:, th_idx], '-o', markersize=2,
                    label=sr.terrain_name, linewidth=1.0)

        ax.set_xlabel('v₀ (m/s)')
        ax.set_ylabel('Final X (m)')
        ax.set_title(f'θ = {theta_target}°')
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc='upper left')

    fig.suptitle('Distance vs Speed at Fixed Launch Angles', fontsize=12)
    fig.tight_layout()

    path = os.path.join(output_dir, 'cross_section_angle.png')
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 5: Cross-section — fixed speeds, distance vs angle
# ---------------------------------------------------------------------------

def plot_cross_section_speed(sweep_results, fixed_speeds=None, output_dir=None):
    """landing_x vs theta for selected fixed speeds."""
    if fixed_speeds is None:
        fixed_speeds = [25, 35, 45, 55, 65]

    n_speeds = len(fixed_speeds)
    fig, axes = plt.subplots(1, n_speeds, figsize=(3.5 * n_speeds, 4),
                             squeeze=False)

    for ax_idx, v0_target in enumerate(fixed_speeds):
        ax = axes[0, ax_idx]
        for sr in sweep_results:
            v0_idx = np.argmin(np.abs(sr.v0_grid - v0_target))
            ax.plot(sr.theta_grid, sr.final_x[v0_idx, :], '-o', markersize=2,
                    label=sr.terrain_name, linewidth=1.0)

        ax.set_xlabel('θ (deg)')
        ax.set_ylabel('Final X (m)')
        ax.set_title(f'v₀ = {v0_target} m/s')
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc='upper left')

    fig.suptitle('Distance vs Launch Angle at Fixed Speeds', fontsize=12)
    fig.tight_layout()

    path = os.path.join(output_dir, 'cross_section_speed.png')
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 6: Multi-terrain heatmap grid
# ---------------------------------------------------------------------------

def plot_terrain_grid(sweep_results, output_dir):
    """Grid of heatmaps, one per terrain, shared colorbar."""
    n = len(sweep_results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows),
                             squeeze=False, layout='constrained')

    # Find global color range
    vmin = min(np.nanmin(sr.final_x) for sr in sweep_results)
    vmax = max(np.nanmax(sr.final_x) for sr in sweep_results)

    for idx, sr in enumerate(sweep_results):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        X, Y = np.meshgrid(sr.theta_grid, sr.v0_grid)
        Z = np.ma.masked_invalid(sr.final_x)

        im = ax.pcolormesh(X, Y, Z, cmap='viridis', shading='auto',
                           vmin=vmin, vmax=vmax)
        ax.set_xlabel('θ (deg)')
        ax.set_ylabel('v₀ (m/s)')
        ax.set_title(sr.terrain_name)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].set_visible(False)

    cbar = fig.colorbar(im, ax=[ax for row in axes for ax in row if ax.get_visible()],
                        label='Final Position X (m)', shrink=0.9)
    fig.suptitle('Shot → Landing Mapping Across Terrains', fontsize=13)

    path = os.path.join(output_dir, 'terrain_grid.png')
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 7: Terrain profiles
# ---------------------------------------------------------------------------

def plot_terrain_profiles(terrains, output_dir, x_max=400):
    """Line plot of terrain height profiles with typical carry markers."""
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.linspace(0, x_max, 1000)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for terrain, color in zip(terrains, colors):
        y = terrain(x)
        ax.plot(x, y, label=terrain.name, color=color, linewidth=1.2)

    # Typical carry distance markers
    for dist, label in [(150, '150m'), (200, '200m'), (250, '250m'), (300, '300m')]:
        ax.axvline(x=dist, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.text(dist, ax.get_ylim()[1] * 0.95, label, fontsize=7, ha='center',
                color='gray', alpha=0.7)

    ax.set_xlabel('Distance X (m)')
    ax.set_ylabel('Height Y (m)')
    ax.set_title('Terrain Profiles')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, 'terrain_profiles.png')
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 8: Example ball trajectories
# ---------------------------------------------------------------------------

def plot_example_trajectories(terrains, output_dir, shots=None):
    """Plot full ball trajectories overlaid on terrain.

    shots: list of (v0, theta_deg) tuples. Uses sensible defaults if None.
    """
    if shots is None:
        shots = [(30, 25), (45, 20), (55, 15), (65, 10)]

    from simulation import simulate_shot

    n = len(terrains)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                             squeeze=False)

    colors = plt.cm.tab10(np.linspace(0, 1, len(shots)))

    for idx, terrain in enumerate(terrains):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        # Draw terrain
        x_terrain = np.linspace(0, 400, 1000)
        ax.fill_between(x_terrain, terrain(x_terrain), -50,
                        color='tan', alpha=0.4)
        ax.plot(x_terrain, terrain(x_terrain), color='saddlebrown',
                linewidth=1.0, label='Terrain')

        # Simulate and draw trajectories
        for (v0, th), color in zip(shots, colors):
            r = simulate_shot(v0, th, terrain, include_roll=True,
                            record_trajectory=True)
            traj = np.array(r['trajectory'])
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.2,
                    label=f'{v0} m/s, {th}°')
            # Mark landing and final positions
            ax.plot(r['landing_x'], r['landing_y'], 'x', color=color,
                    markersize=6, markeredgewidth=1.5)
            ax.plot(r['final_x'], r['final_y'], 'o', color=color,
                    markersize=5)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(terrain.name)
        ax.set_xlim(0, 350)
        ax.set_ylim(-20, 80)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=6, loc='upper right')

    for idx in range(n, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].set_visible(False)

    fig.suptitle('Example Ball Trajectories', fontsize=13)
    fig.tight_layout()

    path = os.path.join(output_dir, 'trajectories.png')
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Generate all plots
# ---------------------------------------------------------------------------

def generate_all(sweep_results, terrains, output_dir='output'):
    """Generate all standard visualization plots."""
    _ensure_dir(output_dir)
    paths = []

    # Per-terrain heatmaps
    for sr in sweep_results:
        paths.append(plot_landing_heatmap(sr, output_dir))

    # Multi-terrain comparisons
    if len(sweep_results) >= 2:
        paths.append(plot_contour_comparison(sweep_results, output_dir))
        paths.append(plot_difference_map(sweep_results[0], sweep_results[1], output_dir))

    # Cross sections
    paths.append(plot_cross_section_angle(sweep_results, output_dir=output_dir))
    paths.append(plot_cross_section_speed(sweep_results, output_dir=output_dir))

    # Grid overview
    paths.append(plot_terrain_grid(sweep_results, output_dir))

    # Terrain profiles
    paths.append(plot_terrain_profiles(terrains, output_dir))

    # Example trajectories
    paths.append(plot_example_trajectories(terrains, output_dir))

    return paths
