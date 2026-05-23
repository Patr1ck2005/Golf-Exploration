"""Flask-based web UI for 2D golf simulation with HTML5 Canvas frontend."""

import json
import numpy as np
from flask import Flask, request, jsonify

from terrain import get_preset, PRESETS
from simulation import simulate_shot
from constants import HOLE_RADIUS, RADIUS

app = Flask(__name__)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.route('/api/terrain_params/<name>')
def api_terrain_params(name):
    """Return terrain parameters for on-the-fly JS evaluation (infinite terrain)."""
    try:
        terrain = get_preset(name)
    except KeyError:
        return jsonify({'error': f'Unknown terrain: {name}'}), 404

    return jsonify({
        'name': terrain.name,
        'amplitudes': terrain.amplitudes.tolist(),
        'wavenumbers': terrain.wavenumbers.tolist(),
        'phases': terrain.phases.tolist(),
        'offset': float(terrain.offset),
        'base_slope': float(terrain.base_slope),
    })


@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    """Run a single shot simulation and return trajectory."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data'}), 400

    v0 = float(data.get('v0', 50))
    theta = float(data.get('theta', 30))
    cl = float(data.get('cl', 0.20))
    hole_x = data.get('hole_x')
    terrain_name = data.get('terrain', 'links_course')
    start_x = float(data.get('start_x', 0.0))
    start_y = data.get('start_y')

    if hole_x is not None:
        hole_x = float(hole_x)
    if start_y is not None:
        start_y = float(start_y)

    try:
        terrain = get_preset(terrain_name)
    except KeyError:
        return jsonify({'error': f'Unknown terrain: {terrain_name}'}), 404

    result = simulate_shot(
        v0, theta, terrain,
        cl=cl,
        hole_x=hole_x,
        record_trajectory=True,
        start_x=start_x,
        start_y=start_y,
    )

    traj = result.get('trajectory', [])
    # Downsample if too many points for smooth animation
    if len(traj) > 800:
        step = len(traj) // 800 + 1
        traj = traj[::step]
        # Always include last point
        if traj[-1] != result.get('trajectory', [])[-1]:
            traj.append(result.get('trajectory', [])[-1])

    response = {
        'trajectory': [{'x': float(p[0]), 'y': float(p[1])} for p in traj],
        'landing_x': float(result['landing_x']),
        'landing_y': float(result['landing_y']),
        'final_x': float(result['final_x']),
        'final_y': float(result['final_y']),
        'max_height': float(result['max_height']),
        'num_bounces': int(result['num_bounces']),
        'holed': bool(result['holed']),
        'hole_distance': float(result['hole_distance']) if result['hole_distance'] is not None else None,
    }
    return jsonify(response)


# ---------------------------------------------------------------------------
# HTML template (single page, embedded CSS + JS)
# ---------------------------------------------------------------------------

HTML_PAGE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Golf Simulation — Shot State Space</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; }
body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: #eceff1;
  color: #263238;
  display: flex;
  justify-content: center;
  padding: 8px;
}
#container {
  width: min(1100px, 100vw - 16px);
  min-width: 860px;
  background: #fff;
  border: 1px solid #cfd8dc;
  border-radius: 4px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  display: flex;
  flex-direction: column;
  height: calc(100vh - 16px);
}
#hud {
  display: flex; align-items: center; gap: 20px; flex-shrink: 0;
  padding: 8px 16px; background: #fafafa; border-bottom: 1px solid #e0e0e0;
  font-size: 13px; min-height: 36px;
}
.hud-item { display: flex; align-items: center; gap: 6px; white-space: nowrap; }
.hud-label { color: #78909c; font-weight: 500; }
.hud-value { font-weight: 600; font-variant-numeric: tabular-nums; }
.bar-bg {
  width: 60px; height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 4px; transition: width 0.05s; }
.bar-power { background: #43a047; }
#canvas-wrap {
  flex: 1; min-height: 300px;
  position: relative; overflow: hidden;
}
#main-canvas {
  display: block; width: 100%; height: 100%;
  cursor: crosshair; background: #fafbfc;
}
#info-bar {
  display: flex; align-items: center; gap: 16px; flex-shrink: 0;
  padding: 8px 16px; background: #fafafa; border-top: 1px solid #e0e0e0;
  font-size: 13px; min-height: 36px;
}
#mini-canvas {
  border: 1px solid #e0e0e0; border-radius: 3px;
  background: #fff; flex-shrink: 0;
}
#stats { display: flex; gap: 16px; flex-wrap: wrap; flex: 1; }
.stat { display: flex; gap: 4px; white-space: nowrap; }
.stat-label { color: #78909c; }
.stat-value { font-weight: 600; font-variant-numeric: tabular-nums; }
#message {
  font-weight: 600; color: #e53935;
  font-size: 13px; white-space: nowrap;
}
.help { color: #b0bec5; font-size: 11px; white-space: nowrap; }
</style>
</head>
<body>
<div id="container">
  <div id="hud">
    <div class="hud-item">
      <span class="hud-label">Shot</span>
      <span class="hud-value" id="shot-count">1</span>
    </div>
    <div class="hud-item">
      <span class="hud-label">Power</span>
      <div class="bar-bg"><div class="bar-fill bar-power" id="power-bar" style="width:0%"></div></div>
      <span class="hud-value" id="power-val">0</span>
    </div>
    <div class="hud-item">
      <span class="hud-label">Angle</span>
      <span class="hud-value" id="angle-val">--</span>
    </div>
    <div class="hud-item">
      <span class="hud-label">Speed</span>
      <input type="range" id="speed-slider" min="0.25" max="4" step="0.25" value="1"
             style="width:60px;">
      <span class="hud-value" id="speed-val">1x</span>
    </div>
    <div class="help">Drag to aim · Space tee reset · Scroll zoom</div>
  </div>

  <div id="canvas-wrap">
    <canvas id="main-canvas"></canvas>
  </div>

  <div id="info-bar">
    <canvas id="mini-canvas" width="150" height="120"></canvas>
    <div id="stats">
      <div class="stat"><span class="stat-label">Speed:</span><span class="stat-value" id="stat-speed">0.0</span> m/s</div>
      <div class="stat"><span class="stat-label">Height:</span><span class="stat-value" id="stat-height">0.0</span> m</div>
      <div class="stat"><span class="stat-label">To hole:</span><span class="stat-value" id="stat-dist">--</span> m</div>
      <div class="stat"><span class="stat-label">Status:</span><span class="stat-value" id="stat-status">Aiming</span></div>
    </div>
    <div id="message"></div>
  </div>
</div>

<script>
// ======================================================================
// State
// ======================================================================
const state = {
  terrainName: 'links_course',
  terrainParams: null,  // {amplitudes, wavenumbers, phases, offset, base_slope}
  holeX: 200,
  holeY: 0,

  // Ball
  ballX: 0, ballY: 0,
  ballVx: 0, ballVy: 0,
  ballSpeed: 0,

  // Tee (shot origin) — updated after each shot to the final position
  teeX: 0, teeY: 0,

  // Shot
  shotCount: 0,
  trajectory: [],
  trajIndex: 0,

  // Aiming
  dragging: false,
  dragStart: { x: 0, y: 0 },
  dragCurrent: { x: 0, y: 0 },
  aimPower: 0,
  aimTheta: 0,

  // View
  cameraX: 0,
  zoom: 5.0,    // px per meter
  viewWidth: 100, // meters visible
  cameraBias: 0.35,  // latched offset for settle after stop

  // Animation
  animTimer: null,
  animSpeed: 1.0,  // trajectory points per frame (adjustable via slider)
  lastDrawTime: 0,

  // Status
  phase: 'aiming',  // aiming | simulating | animating | resting
  message: '',
};

const canvas = document.getElementById('main-canvas');
const ctx = canvas.getContext('2d');
const miniCanvas = document.getElementById('mini-canvas');
const miniCtx = miniCanvas.getContext('2d');

let W, H;

function resizeCanvas() {
  const wrap = document.getElementById('canvas-wrap');
  const rect = wrap.getBoundingClientRect();
  W = canvas.width = Math.floor(rect.width);
  H = canvas.height = Math.floor(rect.height);
  state.viewWidth = W / state.zoom;
  drawAll();
}
window.addEventListener('resize', resizeCanvas);

// ======================================================================
// Terrain evaluation (on-the-fly, infinite)
// ======================================================================
function terrainHeight(x) {
  const p = state.terrainParams;
  if (!p) return 0;
  let y = p.offset + p.base_slope * x;
  for (let i = 0; i < p.amplitudes.length; i++) {
    y += p.amplitudes[i] * Math.sin(p.wavenumbers[i] * x + p.phases[i]);
  }
  return y;
}

// ======================================================================
// Coordinate transforms
// ======================================================================
function worldToScreen(wx, wy) {
  const sx = (wx - state.cameraX) * state.zoom + 30;
  const sy = H - (wy * state.zoom + 260);
  return [sx, sy];
}

function screenToWorld(sx, sy) {
  const wx = (sx - 30) / state.zoom + state.cameraX;
  const wy = (H - sy - 260) / state.zoom;
  return [wx, wy];
}

// ======================================================================
// Drawing
// ======================================================================
function drawTerrain() {
  if (!state.terrainParams) return;

  // World x range covering the full screen width (pixel 0 to W)
  const [wxLeft] = screenToWorld(0, 0);
  const [wxRight] = screenToWorld(W, 0);
  const step = (wxRight - wxLeft) / W;

  // Draw terrain curve from screen-left to screen-right
  ctx.beginPath();
  let first = true;
  for (let wx = wxLeft; wx <= wxRight + step; wx += step) {
    const wy = terrainHeight(wx);
    const [sx, sy] = worldToScreen(wx, wy);
    if (first) { ctx.moveTo(sx, sy); first = false; }
    else ctx.lineTo(sx, sy);
  }
  // Close polygon: down right edge, across bottom, up left edge
  ctx.lineTo(W, H);
  ctx.lineTo(0, H);
  ctx.closePath();

  ctx.fillStyle = '#d7ccc8';
  ctx.fill();
  ctx.strokeStyle = '#8d6e63';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Re-draw just the terrain line on top
  ctx.beginPath();
  first = true;
  for (let wx = wxLeft; wx <= wxRight + step; wx += step) {
    const wy = terrainHeight(wx);
    const [sx, sy] = worldToScreen(wx, wy);
    if (first) { ctx.moveTo(sx, sy); first = false; }
    else ctx.lineTo(sx, sy);
  }
  ctx.strokeStyle = '#5d4037';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawHole() {
  const [hx, hy] = worldToScreen(state.holeX, state.holeY);
  const r = 8; // pixels

  // Flag pole
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(hx, hy - 20);
  ctx.lineTo(hx, hy);
  ctx.stroke();

  // Flag
  ctx.fillStyle = '#e53935';
  ctx.beginPath();
  ctx.moveTo(hx, hy - 20);
  ctx.lineTo(hx + 14, hy - 14);
  ctx.lineTo(hx, hy - 8);
  ctx.fill();

  // Hole circle
  ctx.fillStyle = '#333';
  ctx.beginPath();
  ctx.arc(hx, hy, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#111';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawBall(x, y, vx, vy) {
  const [bx, by] = worldToScreen(x, y);
  const r = 6;

  // Velocity arrow (while in flight)
  if (Math.abs(vx) > 0.1 || Math.abs(vy) > 0.1) {
    const speed = Math.sqrt(vx * vx + vy * vy);
    const arrowLen = Math.min(40, speed * 0.8);
    const nx = vx / speed;
    const ny = vy / speed;
    ctx.strokeStyle = 'rgba(25,118,210,0.5)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(bx, by);
    ctx.lineTo(bx + nx * arrowLen, by - ny * arrowLen);
    ctx.stroke();
  }

  // Ball body
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.arc(bx, by, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Dimple marks
  ctx.fillStyle = '#ccc';
  ctx.beginPath();
  ctx.arc(bx - 2, by - 2, 1, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(bx + 2, by + 1, 1, 0, Math.PI * 2);
  ctx.fill();
}

function drawAimLine() {
  if (!state.dragging || state.phase !== 'aiming') return;

  const [bx, by] = worldToScreen(state.ballX, state.ballY);

  // Drag vector in screen coords
  const dx = state.dragCurrent.x - state.dragStart.x;
  const dy = state.dragCurrent.y - state.dragStart.y;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 3) return;

  // 1. Draw drag line: from drag-start to cursor (the "pull-back" action)
  ctx.strokeStyle = 'rgba(100,100,100,0.5)';
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(state.dragStart.x, state.dragStart.y);
  ctx.lineTo(state.dragCurrent.x, state.dragCurrent.y);
  ctx.stroke();
  ctx.setLineDash([]);

  // Small circle at drag start
  ctx.fillStyle = 'rgba(100,100,100,0.6)';
  ctx.beginPath();
  ctx.arc(state.dragStart.x, state.dragStart.y, 4, 0, Math.PI * 2);
  ctx.fill();

  // 2. Draw aim line: from ball in shot direction (opposite to drag)
  // Screen-space shot direction = (nx, ny) = (-dx/len, -dy/len)
  // Note: screen y increases downward, world y upward.
  // World shot = (-dx, dy), screen shot = (-dx, -dy) = (nx*len, ny*len).
  const nx = -dx / len;
  const ny = -dy / len;
  const lineLen = state.aimPower / 70 * 180;

  ctx.strokeStyle = 'rgba(255,152,0,0.9)';
  ctx.lineWidth = 2.5;
  ctx.setLineDash([8, 5]);
  ctx.beginPath();
  ctx.moveTo(bx, by);
  ctx.lineTo(bx + nx * lineLen, by + ny * lineLen);
  ctx.stroke();
  ctx.setLineDash([]);

  // Arrowhead at end of aim line
  const tipX = bx + nx * lineLen;
  const tipY = by + ny * lineLen;
  const arrowLen = 10;
  const arrowAngle = Math.atan2(ny, nx);
  ctx.fillStyle = 'rgba(255,152,0,0.9)';
  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(
    tipX - arrowLen * Math.cos(arrowAngle - 0.5),
    tipY + arrowLen * Math.sin(arrowAngle - 0.5)
  );
  ctx.lineTo(
    tipX - arrowLen * Math.cos(arrowAngle + 0.5),
    tipY + arrowLen * Math.sin(arrowAngle + 0.5)
  );
  ctx.closePath();
  ctx.fill();

  // 3. Debug text near cursor
  const debugX = state.dragCurrent.x + 14;
  const debugY = state.dragCurrent.y - 30;
  const displayTheta = state.aimTheta > 90 ? -(180 - state.aimTheta) : state.aimTheta;
  ctx.fillStyle = 'rgba(0,0,0,0.75)';
  ctx.font = 'bold 11px monospace';
  ctx.fillText(`θ=${displayTheta.toFixed(1)}°`, debugX, debugY);
  ctx.fillText(`v0=${state.aimPower.toFixed(0)} m/s`, debugX, debugY + 15);
}

function drawTrajectory() {
  if (state.trajectory.length < 2) return;

  ctx.strokeStyle = 'rgba(33,150,243,0.4)';
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 4]);
  ctx.beginPath();
  for (let i = 0; i <= state.trajIndex && i < state.trajectory.length; i++) {
    const [sx, sy] = worldToScreen(state.trajectory[i].x, state.trajectory[i].y);
    if (i === 0) ctx.moveTo(sx, sy);
    else ctx.lineTo(sx, sy);
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawGrid() {
  ctx.strokeStyle = 'rgba(0,0,0,0.06)';
  ctx.lineWidth = 0.5;
  const step = 25; // meters
  for (let wx = Math.floor(state.cameraX / step) * step; wx < state.cameraX + state.viewWidth; wx += step) {
    const [sx, sy] = worldToScreen(wx, 0);
    ctx.beginPath();
    ctx.moveTo(sx, 0);
    ctx.lineTo(sx, H);
    ctx.stroke();
  }
  // Horizontal grid lines
  for (let wy = 0; wy < 100; wy += step) {
    const [sx, sy] = worldToScreen(0, wy);
    ctx.beginPath();
    ctx.moveTo(0, sy);
    ctx.lineTo(W, sy);
    ctx.stroke();
  }
}

function drawMain() {
  ctx.clearRect(0, 0, W, H);
  drawGrid();
  drawTerrain();
  drawTrajectory();
  drawHole();
  drawAimLine();
  if (state.phase !== 'aiming' || state.trajectory.length > 0) {
    drawBall(state.ballX, state.ballY, state.ballVx, state.ballVy);
  } else {
    drawBall(state.ballX, state.ballY, 0, 0);
  }
}

// ======================================================================
// Minimap
// ======================================================================
function drawMiniMap() {
  const mw = 150, mh = 120;
  miniCtx.clearRect(0, 0, mw, mh);
  miniCtx.fillStyle = '#fafafa';
  miniCtx.fillRect(0, 0, mw, mh);

  const range = 5; // meters
  const scale = mw / (2 * range); // px per meter
  const cx = mw / 2;
  const cy = mh / 2;

  function toMini(wx, wy) {
    return [cx + (wx - state.ballX) * scale, cy - (wy - state.ballY) * scale];
  }

  // Terrain near ball (evaluate on-the-fly)
  miniCtx.beginPath();
  let first = true;
  for (let wx = state.ballX - range; wx <= state.ballX + range; wx += 0.05) {
    const wy = terrainHeight(wx);
    const [mx, my] = toMini(wx, wy);
    if (first) { miniCtx.moveTo(mx, my); first = false; }
    else miniCtx.lineTo(mx, my);
  }
  miniCtx.strokeStyle = '#8d6e63';
  miniCtx.lineWidth = 1.5;
  miniCtx.stroke();

  // Terrain fill below
  miniCtx.lineTo(mw, mh);
  miniCtx.lineTo(0, mh);
  miniCtx.closePath();
  miniCtx.fillStyle = 'rgba(215,204,200,0.4)';
  miniCtx.fill();

  // Ball
  miniCtx.fillStyle = '#fff';
  miniCtx.beginPath();
  miniCtx.arc(cx, cy, 4, 0, Math.PI * 2);
  miniCtx.fill();
  miniCtx.strokeStyle = '#333';
  miniCtx.lineWidth = 1.5;
  miniCtx.stroke();

  // Velocity arrow
  if (state.ballSpeed > 0.5) {
    const nx = state.ballVx / state.ballSpeed;
    const ny = state.ballVy / state.ballSpeed;
    miniCtx.strokeStyle = '#e53935';
    miniCtx.lineWidth = 1.5;
    miniCtx.beginPath();
    miniCtx.moveTo(cx, cy);
    miniCtx.lineTo(cx + nx * 20, cy - ny * 20);
    miniCtx.stroke();
  }

  // Labels
  miniCtx.fillStyle = '#78909c';
  miniCtx.font = '9px sans-serif';
  miniCtx.fillText('v', 4, mh - 4);
}

// ======================================================================
// HUD updates
// ======================================================================
function updateHUD() {
  document.getElementById('shot-count').textContent = state.shotCount;
  document.getElementById('power-val').textContent = state.aimPower.toFixed(0);
  document.getElementById('power-bar').style.width = (state.aimPower / 70 * 100) + '%';
  const displayTheta = state.aimTheta > 90 ? -(180 - state.aimTheta) : state.aimTheta;
  document.getElementById('angle-val').textContent = state.dragging ? displayTheta.toFixed(1) + '°' : '--';
  document.getElementById('stat-speed').textContent = state.ballSpeed.toFixed(1);
  document.getElementById('stat-height').textContent = state.ballY.toFixed(1);
  document.getElementById('stat-dist').textContent = state.holeX !== null
    ? (Math.sqrt((state.ballX - state.holeX)**2 + (state.ballY - state.holeY)**2)).toFixed(1)
    : '--';
  document.getElementById('stat-status').textContent =
    state.phase === 'aiming' ? 'Aiming' :
    state.phase === 'simulating' ? 'Computing...' :
    state.phase === 'animating' ? 'In Flight' :
    state.phase === 'resting' ? (state.message || 'At Rest') : state.phase;
  document.getElementById('message').textContent = state.message;
}

// ======================================================================
// Server communication
// ======================================================================
async function loadTerrain(name) {
  const resp = await fetch('/api/terrain_params/' + name);
  state.terrainParams = await resp.json();
  state.terrainName = name;

  // Set hole height using on-the-fly evaluation
  state.holeY = terrainHeight(state.holeX);

  // Set initial ball position at x=0
  state.ballX = 0;
  state.ballY = terrainHeight(0);
  state.teeX = 0;
  state.teeY = state.ballY;
  state.ballVx = 0;
  state.ballVy = 0;
  state.ballSpeed = 0;
  state.trajectory = [];
  state.trajIndex = 0;
  state.aimPower = 0;
  state.aimTheta = 0;
  state.dragging = false;
  state.phase = 'aiming';
  state.message = '';
  state.cameraX = state.ballX - state.viewWidth * 0.35;
  state.cameraBias = 0.35;
  state.holed = false;
  updateHUD();
  drawAll();
}

async function runSimulation() {
  state.phase = 'simulating';
  updateHUD();

  const resp = await fetch('/api/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      v0: state.aimPower,
      theta: state.aimTheta,
      hole_x: state.holeX,
      terrain: state.terrainName,
      start_x: state.teeX,
      start_y: state.teeY,
    }),
  });
  const data = await resp.json();

  state.trajectory = data.trajectory;
  state.trajIndex = 0;
  state.holed = data.holed;
  state.holeDistance = data.hole_distance;

  // Start animation
  state.phase = 'animating';
  animateShot();
}

function animateShot() {
  if (state.trajIndex >= state.trajectory.length - 1) {
    // Animation complete — ball stays at final position
    state.trajIndex = state.trajectory.length - 1;
    const last = state.trajectory[state.trajIndex];
    state.ballX = last.x;
    state.ballY = last.y;
    state.teeX = last.x;  // next shot starts from here
    state.teeY = last.y;
    state.ballVx = 0;
    state.ballVy = 0;
    state.ballSpeed = 0;
    state.phase = 'resting';
    if (state.holed) {
      state.message = 'HOLED! Space to tee off again.';
    } else {
      const dist = state.holeDistance !== null ? state.holeDistance.toFixed(1) + 'm to hole. ' : '';
      state.message = dist + 'Click & drag to hit again. Space to reset.';
    }
    updateHUD();
    drawAll();
    settleCamera();
    return;
  }

  // Advance trajectory index
  state.trajIndex = Math.min(
    state.trajIndex + Math.ceil(state.animSpeed),
    state.trajectory.length - 1
  );

  const pt = state.trajectory[state.trajIndex];
  const prev = state.trajectory[Math.max(0, state.trajIndex - 1)];

  state.ballX = pt.x;
  state.ballY = pt.y;
  state.ballVx = (pt.x - prev.x) / 0.016; // approx velocity
  state.ballVy = (pt.y - prev.y) / 0.016;
  state.ballSpeed = Math.sqrt(state.ballVx**2 + state.ballVy**2);

  // Smooth camera bias — doesn't flip instantly on direction reversal
  const vx = state.ballVx;
  const targetBias = vx > 3 ? 0.15 : (vx < -3 ? 0.65 : state.cameraBias);
  state.cameraBias += (targetBias - state.cameraBias) * 0.006;
  // Simple lerp toward target
  const targetX = state.ballX - state.viewWidth * state.cameraBias;
  state.cameraX += (targetX - state.cameraX) * 0.015;

  updateHUD();
  drawAll();

  state.animTimer = requestAnimationFrame(animateShot);
}

function settleCamera() {
  // Simple lerp to final position with latched bias
  const targetX = state.ballX - state.viewWidth * state.cameraBias;
  state.cameraX += (targetX - state.cameraX) * 0.01;
  if (Math.abs(targetX - state.cameraX) < 0.05) return;
  drawAll();
  requestAnimationFrame(settleCamera);
}

function drawAll() {
  drawMain();
  drawMiniMap();
}

function drawAllThrottled() {
  const now = performance.now();
  if (now - state.lastDrawTime < 33) return; // ~30fps
  state.lastDrawTime = now;
  drawAll();
}

// ======================================================================
// Input handling
// ======================================================================
canvas.addEventListener('mousedown', (e) => {
  if (state.phase === 'animating' || state.phase === 'simulating') return;
  // Start aiming from current ball position (no reset to tee)
  state.phase = 'aiming';
  state.dragging = true;
  state.dragStart = { x: e.offsetX, y: e.offsetY };
  state.dragCurrent = { x: e.offsetX, y: e.offsetY };
  state.aimPower = 0;
  state.message = '';
  // Set tee to current ball position for this shot
  state.teeX = state.ballX;
  state.teeY = state.ballY;
  updateHUD();
  drawAll();
});

// Mousemove on document so drag continues when mouse leaves canvas
document.addEventListener('mousemove', (e) => {
  if (!state.dragging) return;

  // Get mouse position relative to canvas
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left;
  const sy = e.clientY - rect.top;
  state.dragCurrent = { x: sx, y: sy };

  // Compute aim from drag (opposite direction)
  const dx = state.dragCurrent.x - state.dragStart.x;
  const dy = state.dragCurrent.y - state.dragStart.y;
  const len = Math.sqrt(dx * dx + dy * dy);

  if (len > 3) {
    // World-space shot direction: opposite to screen drag
    // screen drag (dx,dy) -> world shot = (-dx, dy)  [y flipped]
    const wx = -dx;
    const wy = dy;
    // atan2(wy, wx): wx>0 → 0-90° (right), wx<0 → 90-180° (left)
    state.aimTheta = Math.atan2(wy, wx) * 180 / Math.PI;
    // Allow full range: 5° (flat right) to 175° (flat left)
    state.aimTheta = Math.max(5, Math.min(175, state.aimTheta));
    state.aimPower = Math.min(70, len / 5);
    state.aimPower = Math.max(5, state.aimPower);
  }

  updateHUD();
  drawAllThrottled();
});

canvas.addEventListener('mouseup', (e) => {
  if (!state.dragging) return;
  state.dragging = false;

  if (state.aimPower > 5) {
    console.log('Shot:', {v0: state.aimPower.toFixed(1), theta: state.aimTheta.toFixed(1), hole_x: state.holeX, terrain: state.terrainName});
    state.shotCount++;
    runSimulation();
  }

  updateHUD();
  drawAll();
});

// Catch mouseup anywhere on document so drag doesn't freeze if mouse leaves canvas
document.addEventListener('mouseup', (e) => {
  if (state.dragging) {
    if (state.aimPower > 5) {
      console.log('Shot (document):', {v0: state.aimPower.toFixed(1), theta: state.aimTheta.toFixed(1), start_x: state.teeX, start_y: state.teeY});
      state.shotCount++;
      runSimulation();
    }
    state.dragging = false;
    updateHUD();
    drawAll();
  }
});

canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  state.zoom *= (1 - e.deltaY * 0.001);
  state.zoom = Math.max(1, Math.min(20, state.zoom));
  state.viewWidth = W / state.zoom;
  drawAll();
});

document.getElementById('speed-slider').addEventListener('input', (e) => {
  state.animSpeed = parseFloat(e.target.value);
  document.getElementById('speed-val').textContent = state.animSpeed.toFixed(2) + 'x';
});

document.addEventListener('keydown', (e) => {
  if (e.key === ' ') {
    e.preventDefault();
    resetShot();
  }
});

// ======================================================================
// Shot lifecycle
// ======================================================================
function resetShot() {
  if (state.animTimer) cancelAnimationFrame(state.animTimer);
  // Reset to original tee at x=0
  state.ballX = 0;
  state.ballY = terrainHeight(0);
  state.teeX = 0;
  state.teeY = state.ballY;
  state.ballVx = 0;
  state.ballVy = 0;
  state.ballSpeed = 0;
  state.trajectory = [];
  state.trajIndex = 0;
  state.aimPower = 0;
  state.aimTheta = 0;
  state.dragging = false;
  state.phase = 'aiming';
  state.message = '';
  state.cameraX = state.ballX - state.viewWidth * 0.35;
  state.cameraBias = 0.35;
  state.holed = false;
  updateHUD();
  drawAll();
}

// ======================================================================
// Startup
// ======================================================================
async function init() {
  // Read URL query parameters: ?terrain=links_course&hole=180
  const params = new URLSearchParams(window.location.search);
  if (params.get('terrain')) {
    state.terrainName = params.get('terrain');
  }
  if (params.get('hole')) {
    state.holeX = parseFloat(params.get('hole'));
  }
  await loadTerrain(state.terrainName);
  resizeCanvas();
  state.cameraX = state.ballX - state.viewWidth * 0.35;
  state.cameraBias = 0.35;
  drawAll();
  updateHUD();
}

init();
</script>
</body>
</html>
'''


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return HTML_PAGE


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def start_web(host='127.0.0.1', port=5000, debug=False):
    """Start the Flask web server."""
    print(f"\nGolf Simulation Web UI")
    print(f"Open http://{host}:{port} in your browser")
    print(f"Drag mouse to aim & shoot | R/E adjust spin | Space reset | Scroll zoom\n")
    app.run(host=host, port=port, debug=debug)
