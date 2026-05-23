"""Forces and RK4 integrator for 2D golf ballistics."""

import numpy as np
from constants import G, K_AERO, CD, CL


def forces(state, cd=CD, cl=CL):
    """Compute acceleration (ax, ay) for state = [x, y, vx, vy].

    Forces: gravity + quadratic drag + Magnus lift (backspin).
    """
    vx, vy = state[2], state[3]
    speed = np.sqrt(vx * vx + vy * vy)

    if speed < 1e-12:
        return np.array([0.0, -G])

    # aero = a_drag + a_magnus
    # a_drag = -K_AERO * cd * (vx, vy)             opposes velocity
    # a_magnus = K_AERO * cl * (-vy, vx)           lift from backspin (CCW rotation)
    # a_total = (-K_AERO * (cd * vx + cl * vy),
    #            -K_AERO * (cd * vy - cl * vx))
    factor = -K_AERO
    ax = factor * (cd * vx + cl * vy)
    ay = factor * (cd * vy - cl * vx) - G

    return np.array([ax, ay])


def rk4_step(state, dt, force_func=forces):
    """Advance state by dt using classical RK4.

    state = [x, y, vx, vy]
    Returns new state.
    """
    # k1
    a1 = force_func(state)
    k1v = np.array([state[2], state[3], a1[0], a1[1]])

    # k2
    s2 = state + 0.5 * dt * k1v
    a2 = force_func(s2)
    k2v = np.array([s2[2], s2[3], a2[0], a2[1]])

    # k3
    s3 = state + 0.5 * dt * k2v
    a3 = force_func(s3)
    k3v = np.array([s3[2], s3[3], a3[0], a3[1]])

    # k4
    s4 = state + dt * k3v
    a4 = force_func(s4)
    k4v = np.array([s4[2], s4[3], a4[0], a4[1]])

    return state + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)


def integrate(state0, dt, max_time, stop_fn):
    """Integrate state0 forward, yielding (t, state) each step.

    Stops when stop_fn(t, state_prev, state_curr) returns True
    or when t exceeds max_time.

    Args:
        state0: initial [x, y, vx, vy]
        dt: time step
        max_time: maximum simulation time
        stop_fn: callable(t, state_prev, state) -> bool

    Yields:
        (t, state) tuples
    """
    t = 0.0
    state = np.array(state0, dtype=np.float64)
    yield t, state.copy()

    while t < max_time:
        t += dt
        state_prev = state.copy()
        state = rk4_step(state, dt)

        if stop_fn(t - dt, state_prev, state):
            yield t, state.copy()
            return

        yield t, state.copy()
