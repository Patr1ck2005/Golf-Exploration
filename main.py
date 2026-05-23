"""2D Golf Simulation — State Space Visualization.

Entry point: runs parameter sweeps over selected terrains and generates
all visualization plots. Use --web for interactive web UI.
"""

import argparse
import sys
import time

from terrain import get_preset, PRESETS


def main():
    parser = argparse.ArgumentParser(
        description='2D Golf Simulation — Shot Parameter to Landing Position Mapping'
    )
    parser.add_argument('--web', action='store_true',
                        help='Launch interactive web UI (Flask + Canvas)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port for web UI (default: 5000)')
    parser.add_argument('--terrain', type=str, default='all',
                        help='Terrain presets, comma-separated, or "all" (default: all)')
    parser.add_argument('--resolution', type=int, default=50,
                        help='Grid resolution N for NxN sweep (default: 50)')
    parser.add_argument('--v0-range', type=str, default='20,70',
                        help='Min,max initial speed in m/s (default: 20,70)')
    parser.add_argument('--theta-range', type=str, default='5,60',
                        help='Min,max launch angle in degrees (default: 5,60)')
    parser.add_argument('--dt', type=float, default=0.01,
                        help='RK4 integration time step (default: 0.01)')
    parser.add_argument('--no-roll', action='store_true',
                        help='Disable rolling phase (stop at first impact)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory for output plots (default: output/)')
    args = parser.parse_args()

    # --- Web UI mode ---
    if args.web:
        from web_ui import start_web
        start_web(port=args.port)
        return

    # --- Sweep mode ---
    # Force non-interactive backend before importing visualization
    import matplotlib
    matplotlib.use('Agg')
    from sweep import parameter_sweep
    from visualization import generate_all

    if args.terrain == 'all':
        terrain_names = list(PRESETS.keys())
    else:
        terrain_names = [t.strip() for t in args.terrain.split(',')]

    terrains = []
    for name in terrain_names:
        try:
            terrains.append(get_preset(name))
        except KeyError as e:
            print(f"Error: {e}")
            sys.exit(1)

    print(f"Selected terrains: {', '.join(t.name for t in terrains)}")

    # --- Parse ranges ---
    v0_range = tuple(map(float, args.v0_range.split(',')))
    theta_range = tuple(map(float, args.theta_range.split(',')))
    include_roll = not args.no_roll

    print(f"Speed range: {v0_range[0]}–{v0_range[1]} m/s")
    print(f"Angle range: {theta_range[0]}–{theta_range[1]}°")
    print(f"Grid resolution: {args.resolution}×{args.resolution} = {args.resolution**2} shots/terrain")
    print(f"Rolling: {'enabled' if include_roll else 'disabled'}")
    print()

    # --- Run sweeps ---
    total_start = time.perf_counter()
    sweep_results = []

    for terrain in terrains:
        print(f"Simulating [{terrain.name}]...")
        t0 = time.perf_counter()
        sr = parameter_sweep(
            terrain,
            v0_range=v0_range,
            theta_range=theta_range,
            resolution=args.resolution,
            dt=args.dt,
            include_roll=include_roll,
            verbose=True,
        )
        elapsed = time.perf_counter() - t0
        print(f"  Done in {elapsed:.1f}s\n")
        sweep_results.append(sr)

    total_elapsed = time.perf_counter() - total_start
    print(f"All sweeps complete: {total_elapsed:.1f}s total")

    # --- Generate visualizations ---
    print(f"\nGenerating plots in {args.output_dir}/ ...")
    t0 = time.perf_counter()
    paths = generate_all(sweep_results, terrains, output_dir=args.output_dir)
    plot_elapsed = time.perf_counter() - t0

    print(f"Generated {len(paths)} plot(s) in {plot_elapsed:.1f}s:")
    for p in paths:
        print(f"  {p}")

    # --- Summary statistics ---
    print("\n=== Summary ===")
    for sr in sweep_results:
        flat_idx = sr.final_x.argmax()
        best_v0_idx, best_th_idx = np.unravel_index(flat_idx, sr.final_x.shape)
        best_v0 = sr.v0_grid[best_v0_idx]
        best_th = sr.theta_grid[best_th_idx]
        print(f"  {sr.terrain_name}: max distance = {sr.final_x.max():.1f}m "
              f"(v0={best_v0:.1f} m/s, theta={best_th:.1f} deg)")


if __name__ == '__main__':
    import numpy as np
    main()
