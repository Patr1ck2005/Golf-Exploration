"""Physical constants and ball properties for 2D golf simulation."""

import numpy as np

# Gravity
G = 9.81  # m/s^2

# Air density (sea level, 15°C)
RHO = 1.225  # kg/m^3

# Golf ball properties (regulation)
MASS = 0.0459  # kg
RADIUS = 0.02135  # m
AREA = np.pi * RADIUS ** 2  # cross-sectional area, m^2

# Aerodynamic coefficients
CD = 0.25  # drag coefficient
CL = 0.20  # lift coefficient (backspin; scales linearly with spin rate)

# Combined drag/lift constant: k = 0.5 * rho * A / m
K_AERO = 0.5 * RHO * AREA / MASS

# Integration defaults
DT_DEFAULT = 0.01  # s, RK4 step size
MAX_FLIGHT_TIME = 60.0  # s, safety cap

# Collision / bounce
RESTITUTION = 0.4  # coefficient of restitution (grass)
TANGENTIAL_FRICTION = 0.10  # velocity reduction along surface on bounce

# Hole (standard golf hole: 4.25 inches diameter)
HOLE_RADIUS = 0.053975  # m (4.25" / 2)

# Rolling
ROLL_THRESHOLD = 2.0  # m/s, below which ball transitions to rolling
STOP_THRESHOLD = 0.1  # m/s, below which ball stops
ROLLING_FRICTION = 0.1  # rolling resistance coefficient
MAX_ROLL_DISTANCE = 50.0  # m, safety cap on rolling distance
DT_ROLL = 0.005  # s, rolling integration step
