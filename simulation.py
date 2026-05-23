"""Single-shot golf simulation: flight, terrain collision, bounce, rolling."""

import numpy as np
from physics import forces, rk4_step
from terrain import Terrain
from constants import (
    G, CD, CL, DT_DEFAULT, MAX_FLIGHT_TIME,
    RESTITUTION, TANGENTIAL_FRICTION,
    ROLL_THRESHOLD, STOP_THRESHOLD, ROLLING_FRICTION,
    MAX_ROLL_DISTANCE, DT_ROLL, RADIUS,
    HOLE_RADIUS,
)


def _bisect_impact(state_prev, state_curr, terrain, dt_step, force_fn):
    """Find precise impact state via bisection with linear interpolation.

    Returns state at the terrain crossing point (ball just touching surface).
    """
    t_lo, t_hi = 0.0, dt_step
    clearance_hi = state_curr[1] - float(terrain(state_curr[0]))

    for _ in range(15):
        t_mid = 0.5 * (t_lo + t_hi)
        alpha = t_mid / dt_step
        s_mid = state_prev + alpha * (state_curr - state_prev)
        clearance = s_mid[1] - float(terrain(s_mid[0]))

        if clearance > 0:
            t_lo = t_mid
        else:
            t_hi = t_mid

    alpha_final = t_lo / dt_step
    return state_prev + alpha_final * (state_curr - state_prev)


def apply_bounce(state, terrain):
    """Apply coefficient of restitution and tangential friction at impact.

    state: [x, y, vx, vy] at impact point.
    Returns new state after bounce.
    """
    x, y, vx, vy = state
    nx, ny = terrain.normal(x)  # upward-pointing normal

    # Decompose velocity into normal and tangential components
    vn = vx * nx + vy * ny          # normal component (into terrain, negative)
    vt_x = vx - vn * nx
    vt_y = vy - vn * ny

    # Reverse and damp normal component; damp tangential component
    if vn < 0:  # ball moving into terrain
        vn_new = -RESTITUTION * vn
    else:
        vn_new = vn  # shouldn't happen, but be safe

    vt_x *= (1.0 - TANGENTIAL_FRICTION)
    vt_y *= (1.0 - TANGENTIAL_FRICTION)

    # Recombine
    vx_new = vt_x + vn_new * nx
    vy_new = vt_y + vn_new * ny

    # Nudge ball above terrain to prevent re-collision
    terrain_y = float(terrain(x))
    new_state = np.array([x, max(terrain_y + 1e-6, y), vx_new, vy_new])

    return new_state


def simulate_rolling(x0, y0, vx0, vy0, terrain, hole_x=None, record_path=False):
    """Simulate rolling along terrain surface until ball stops.

    Returns (final_x, final_y, holed, path) where path is a list of (x, y)
    if record_path=True, else an empty list.
    """
    x = float(x0)
    vx, vy = float(vx0), float(vy0)
    path = []  # don't duplicate the impact point; it's already in the trajectory

    # Project initial velocity onto terrain tangent
    s = float(terrain.slope(x))
    theta = np.arctan(s)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    v_tan = vx * cos_t + vy * sin_t

    total_dist = 0.0
    stopped_frames = 0
    step_count = 0

    while total_dist < MAX_ROLL_DISTANCE:
        step_count += 1
        s = float(terrain.slope(x))
        theta = np.arctan(s)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        speed = abs(v_tan)
        if speed < STOP_THRESHOLD:
            max_static_accel = ROLLING_FRICTION * G * abs(cos_t)
            slope_accel = abs(G * sin_t)
            if slope_accel <= max_static_accel + 1e-6:
                stopped_frames += 1
                if stopped_frames >= 5:
                    break
            else:
                v_tan = -np.sign(sin_t) * STOP_THRESHOLD * 2
                stopped_frames = 0
        else:
            stopped_frames = 0

        # Hole detection: ball rolls into the terrain gap
        if _in_hole_gap(x, hole_x):
            if record_path:
                path.append((x, float(terrain(x))))
            return x, float(terrain(x)), True, path

        a_gravity = -G * sin_t
        a_friction = -ROLLING_FRICTION * G * abs(cos_t) * np.sign(v_tan)
        a_tan = a_gravity + a_friction

        v_tan_new = v_tan + a_tan * DT_ROLL

        if v_tan * v_tan_new < 0 and abs(v_tan) < 0.5:
            slope_accel = abs(G * sin_t)
            max_friction_accel = ROLLING_FRICTION * G * abs(cos_t)
            if slope_accel <= max_friction_accel + 1e-6:
                break
            else:
                v_tan_new = -np.sign(sin_t) * 0.01

        v_tan = v_tan_new
        ds = v_tan * DT_ROLL
        dx = ds * cos_t
        x += dx
        total_dist += abs(ds)

        # Record every 2nd step → 2*0.005=0.01s, matching flight dt exactly
        if record_path and step_count % 2 == 0:
            path.append((x, float(terrain(x))))

        if x < 0 or x > 1000:
            break

    y = float(terrain(x))
    if record_path:
        path.append((x, y))
    return x, y, False, path


def _in_hole_gap(x, hole_x):
    """Return True if x is within the hole's terrain gap.

    The hole is modelled as a narrow vertical cliff:
    for |x - hole_x| < HOLE_RADIUS the terrain is absent (infinite depth).
    Ball entering this gap while rolling or landing falls in.
    """
    return hole_x is not None and abs(x - hole_x) < HOLE_RADIUS


def simulate_shot(v0, theta_deg, terrain, tee_height=0.0, dt=None,
                  cd=None, cl=None, include_roll=True, record_trajectory=False,
                  hole_x=None, start_x=0.0, start_y=None):
    """Simulate a single golf shot from tee to rest.

    Args:
        v0: initial speed (m/s)
        theta_deg: launch angle (degrees from horizontal)
        terrain: Terrain instance
        tee_height: height of tee above terrain at x=0 (m)
        dt: RK4 time step (default DT_DEFAULT)
        cd: drag coefficient override
        cl: lift coefficient override
        include_roll: whether to simulate rolling after landing
        record_trajectory: if True, store full flight path (for plots)
        hole_x: x-position of hole on terrain (None = no hole)

    Returns dict with keys:
        landing_x, landing_y: first impact position
        final_x, final_y: resting position after roll
        max_height: peak altitude during flight
        num_bounces: number of bounces before rolling
        total_distance: final_x (alias)
        holed: True if ball entered the hole
        hole_distance: final distance from hole center
        trajectory: list of (x, y) points (only if record_trajectory=True)
    """
    if dt is None:
        dt = DT_DEFAULT

    theta = np.radians(theta_deg)
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)

    x0 = float(start_x)
    y0 = float(start_y) if start_y is not None else float(terrain(x0)) + tee_height

    cd_val = cd if cd is not None else CD
    cl_val = cl if cl is not None else CL
    force_fn = lambda s: forces(s, cd=cd_val, cl=cl_val)

    state = np.array([x0, y0, vx0, vy0], dtype=np.float64)
    t = 0.0
    holed = False

    trajectory = [(x0, y0)] if record_trajectory else None
    flight_max_height = y0
    landing_x = landing_y = None
    num_bounces = 0

    # --- Flight & bounce loop ---
    while t < MAX_FLIGHT_TIME and num_bounces < 5:
        t += dt
        state_prev = state.copy()
        state = rk4_step(state, dt, force_fn)

        # Track max height during flight
        if state[1] > flight_max_height:
            flight_max_height = float(state[1])

        clearance_curr = state[1] - float(terrain(state[0]))

        # Only record flight point if above terrain (underground points create artifacts)
        if record_trajectory and clearance_curr > 0:
            trajectory.append((float(state[0]), float(state[1])))

        if clearance_curr <= 0:
            # Collision detected — bisect to find impact
            impact = _bisect_impact(state_prev, state, terrain, dt, force_fn)

            # Check if impact is within the hole gap (narrow cliff model)
            if _in_hole_gap(float(impact[0]), hole_x):
                holed = True
                if record_trajectory:
                    trajectory.append((float(impact[0]), float(impact[1])))
                break

            if landing_x is None:
                landing_x = float(impact[0])
                landing_y = float(impact[1])

            if record_trajectory:
                trajectory.append((float(impact[0]), float(impact[1])))

            # Bounce
            state = apply_bounce(impact, terrain)
            speed = np.sqrt(state[2]**2 + state[3]**2)
            num_bounces += 1

            # Check if ball has enough vertical speed to actually leave terrain
            nx, ny = terrain.normal(state[0])
            vn = state[2] * nx + state[3] * ny  # normal velocity (should be positive = upward)

            if speed < ROLL_THRESHOLD or vn < 1.0 or num_bounces >= 5:
                break

            # Re-nudge above terrain for next flight phase
            ty = float(terrain(state[0]))
            if state[1] <= ty:
                state[1] = ty + 1e-6

    # --- Rolling phase ---
    x_rest = float(state[0])
    y_rest = float(state[1])
    roll_path = []

    if include_roll and not holed:
        speed = np.sqrt(state[2]**2 + state[3]**2)
        if speed > STOP_THRESHOLD:
            x_rest, y_rest, holed, roll_path = simulate_rolling(
                state[0], state[1], state[2], state[3], terrain,
                hole_x=hole_x,
                record_path=record_trajectory,
            )
            if record_trajectory and roll_path:
                trajectory.extend(roll_path)
        else:
            y_rest = float(terrain(x_rest))

    if record_trajectory and not (include_roll and roll_path):
        trajectory.append((x_rest, y_rest))

    # Distance to hole (horizontal distance to the cliff gap)
    hole_dist = abs(x_rest - hole_x) if hole_x is not None else None

    result = {
        'landing_x': landing_x if landing_x is not None else x_rest,
        'landing_y': landing_y if landing_y is not None else y_rest,
        'final_x': x_rest,
        'final_y': y_rest,
        'max_height': flight_max_height,
        'num_bounces': num_bounces,
        'total_distance': x_rest,
        'holed': holed,
        'hole_distance': hole_dist,
    }
    if record_trajectory:
        result['trajectory'] = trajectory
    return result
