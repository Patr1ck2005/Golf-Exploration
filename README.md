# Golf Simulation — Shot State Space Visualization

2D golf physics simulation with procedural sand-dune terrain, interactive web UI, and batch scientific visualization. Core goal: visualize the mapping from shot parameters `(v0, θ)` to ball landing position, and how this mapping depends on the terrain.

## Quick Start

```bash
pip install flask numpy matplotlib scipy
python main.py --web
```

Open `http://127.0.0.1:5000` in browser.

## Usage

### Interactive Web UI

```bash
python main.py --web                           # default terrain, hole at x=200m
python main.py --web --terrain steep_dunes     # specific terrain
python main.py --web --hole 180                # hole position
python main.py --web --port 8080               # custom port
```

**Controls**:
- Drag mouse anywhere → aim (drag direction = pull-back, opposite = shot)
- Release → fire
- Space → reset to tee (x=0)
- Scroll wheel → zoom
- Speed slider in HUD → animation playback speed

### Batch Sweep Mode

```bash
python main.py                                  # all terrains, 50x50 grid
python main.py --terrain flat,valley --resolution 100
python main.py --v0-range 10,80 --theta-range 2,70
```

Generates PNG figures in `output/`: heatmaps, contour comparisons, cross-sections, trajectory plots.

## Physics Model

2D plane (x horizontal, y vertical). RK4 integration at dt=0.01s.

| Force | Formula |
|-------|---------|
| Gravity | `a_y = -g` |
| Drag | `a_d = -k·|v|·Cd·(vx, vy)` |
| Magnus lift | `a_m = -k·|v|·Cl·(-vy, vx)` (backspin → upward lift) |

Collision: bisection to find terrain impact, restitution 0.4, tangential friction 0.1. Rolling: constrained to terrain surface with rolling friction.

## Project Structure

```
Golf/
├── main.py             # Entry point (--web or sweep mode)
├── constants.py        # Physical constants
├── physics.py          # Forces, RK4 integrator
├── terrain.py          # Terrain class, 6 presets (sum of sinusoids)
├── simulation.py       # Single shot: flight, bounce, rolling, hole detection
├── sweep.py            # Parameter grid sweep engine
├── visualization.py    # Matplotlib static plots
├── web_ui.py           # Flask server + embedded HTML/Canvas frontend
└── output/             # Generated figures
```

## Terrain Presets

Infinite procedural terrain via `y = offset + slope·x + Σ Ai·sin(ki·x + φi)`.

| Name | Description |
|------|-------------|
| flat | Reference baseline |
| gentle_dunes | Low rolling hills (±21m) |
| steep_dunes | High dunes (±33m) |
| uphill | Steady climb + dunes |
| valley | Deep valley depression |
| links_course | Irregular multi-scale dunes (±36m) |

## Hole Model

Narrow vertical cliff: terrain gap of width `2 × hole_radius` (4.25" diameter). Ball is holed when rolling into the gap or landing directly in it. Flying over at altitude does not count.
