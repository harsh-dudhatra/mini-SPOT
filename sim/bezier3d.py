"""
3D Bezier gait from D^2-GMBC (Rahme et al. 2020, arXiv 2010.12070) for Mini-SPOT, with a
closed-form leg IK (sim/InverseKinematics.py) so it is fast enough to run inside RL training.

  command  (vx, vy, wz)  body-frame m/s and rad/s     -> forward/back, strafe, turn
       |                                                  (one gait for all directions)
       v
  per leg: phase S in [0, 2)  (stance 0-1, swing 1-2, trot timing)
       |
       |  translation curve  (q, z) = Upsilon(S, L_span, psi, delta)   paper Eq. 13, Table V
       |  yaw curve          (q, -) = Upsilon(S, omega_bar, ...)        paper Eq. 4-11
       v
  foot target f_xyz = f_stand + translation (rotated by rho) + yaw (tangent to the turn circle)
       |                                                               (+ RL residual delta_f)
       v
  IK (closed form, per leg): f_xyz -> (hip, thigh, knee) angles  -> MuJoCo position servos

The planar curve Upsilon (12-point Bezier swing + sine stance) is defined below (moved here from
the old 3dbezierhelpers.py demo, now in archive/). psi = foot clearance height, delta = virtual ground penetration depth:
the two curve parameters the paper's policy sets.

Yaw: the paper builds each foot's turning direction from its previous position (Eq. 6-10). Here
the foot moves tangent to the circle through its stand position, which is what those equations
aim for (four-wheel-steering turn about the body centre) and has no state.
"""

import math

import numpy as np

LEG_NAMES = ["FR", "FL", "RR", "RL"]
TROT_LAGS = {"FR": 0.0, "FL": 0.5, "RR": 0.5, "RL": 0.0}     # diagonal pairs (paper Eq. 16)
STAND_ANGLES = (0.0, 0.687, -1.0)                           # (hip, thigh, knee) standing pose

PSI_DEFAULT = 0.04       # m, swing clearance height
DELTA_DEFAULT = 0.005    # m, stance penetration depth


# ---------------------------------------------------------------------------
# Eq. (14) + Table V: Bezier curve for the SWING phase
# ---------------------------------------------------------------------------
def _bernstein(n: int, k: int, s: float) -> float:
    # B_k^n(s) = C(n,k) (1-s)^(n-k) s^k   (paper's Eq. 14 lost the s^k exponent
    # in typesetting; this is the standard Bernstein polynomial)
    return math.comb(n, k) * (1.0 - s) ** (n - k) * s ** k


def _control_points(tau: float, psi: float) -> np.ndarray:
    """12 control points c_k(tau, psi) in the (q, z) plane, Table V.
    tau = step length, psi = clearance height  (psi is RL-controlled)."""
    return np.array([
        (-tau,        0.0),        # c0
        (-1.4 * tau,  0.0),        # c1
        (-1.5 * tau,  0.9 * psi),  # c2
        (-1.5 * tau,  0.9 * psi),  # c3
        (-1.5 * tau,  0.9 * psi),  # c4
        (0.0,         0.9 * psi),  # c5
        (0.0,         0.9 * psi),  # c6
        (0.0,         1.1 * psi),  # c7
        (+1.5 * tau,  1.1 * psi),  # c8  (sign: see note below)
        (+1.5 * tau,  1.1 * psi),  # c9
        (+1.4 * tau,  0.0),        # c10
        (+tau,        0.0),        # c11
    ])
    # NOTE: the extracted Table V shows negative q for c8..c10. The swing has to
    # travel from -tau (c0) to +tau (c11), so the symmetric positive values are
    # used here (matches the original MIT-Cheetah curve of Hyun et al.).


def _swing(s_swing: float, tau: float, psi: float):
    """Swing: Bezier curve, s_swing in [0,1] -> (q, z)."""
    pts = _control_points(tau, psi)
    n = len(pts) - 1
    q = z = 0.0
    for k in range(n + 1):
        b = _bernstein(n, k, s_swing)
        q += b * pts[k, 0]
        z += b * pts[k, 1]
    return q, z
#maybe using matrix multiplication would be faster, but this is only called 4 times per step.

def _stance(s_stance: float, tau: float, delta: float):
    """Stance: sinusoidal dip, s_stance in [0,1) -> (q, z).
    Foot slides backwards from +tau to -tau while dipping by `delta`
    (delta is RL-controlled).  z = 0 at both ends so it joins the swing curve."""
    u = 1.0 - 2.0 * s_stance                 # +1 -> -1 over the stance
    q = tau * u
    z = -delta * math.cos(0.5 * math.pi * u)  # 0 at u=+-1, -delta at u=0
    return q, z


def trajectory_generator(S: float, tau: float, psi: float, delta: float):
    """Upsilon(S, tau): phase S in [0,2) -> planar point (q, z).
       S in [0,1): stance (sine curve)   S in [1,2): swing (Bezier curve)."""
    if S < 1.0:
        return _stance(S, tau, delta)
    return _swing(S - 1.0, tau, psi)


class Bezier3DGait:
    """Command-driven foot trajectories (paper Algorithm 3). Never touches the MuJoCo model:
    the legs (LegIK, one per LEG_NAMES entry) come from sim/mujoco_driver.MujocoDriver, which
    also turns foot targets into actuator ctrl (MujocoDriver.joint_targets)."""

    def __init__(self, legs, t_swing=0.2, stance_to_swing=1.0):   # paper: T_swing = 0.2 s
        self.t_swing = t_swing
        self.t_stance = stance_to_swing * t_swing
        self.t_stride = self.t_swing + self.t_stance
        self.stance_frac = self.t_stance / self.t_stride
        self.legs = list(legs)
        self.lags = np.array([TROT_LAGS[leg] for leg in LEG_NAMES])
        self.f_stand = np.array([leg.fk(STAND_ANGLES) for leg in self.legs])     # (4, 3)

    def leg_phase(self, phase, i):
        """Stride phase in [0, 1) -> paper's S in [0, 2): stance 0-1, swing 1-2."""
        u = (phase + self.lags[i]) % 1.0
        if u < self.stance_frac:
            return u / self.stance_frac
        return 1.0 + (u - self.stance_frac) / (1.0 - self.stance_frac)

    def command_to_curve(self, cmd):
        """(vx, vy, wz) -> (rho, L_span, omega_bar) for the planar curves."""
        vx, vy, wz = cmd
        rho = np.arctan2(vy, vx)
        L_span = np.hypot(vx, vy) * self.t_stance / 2.0       # half stride: covered in one stance
        return rho, L_span, wz * self.t_stance / 2.0          # omega_bar in rad (x radius below)

    def foot_targets(self, phase, cmd, psi=PSI_DEFAULT, delta=DELTA_DEFAULT, residual=None):
        rho, L_span, yaw_half = self.command_to_curve(cmd)
        f = self.f_stand.copy()
        for i in range(4):
            S = self.leg_phase(phase, i)
            q_tr, z = trajectory_generator(S, L_span, psi, delta)
            fx, fy = self.f_stand[i, :2]
            radius = np.hypot(fx, fy)
            q_yaw, _ = trajectory_generator(S, yaw_half * radius, psi, delta)
            tangent = np.array([-fy, fx]) / radius            # counter-clockwise turn direction
            f[i, :2] += q_tr * np.array([np.cos(rho), np.sin(rho)]) + q_yaw * tangent
            f[i, 2] += z
        if residual is not None:
            f += residual
        return f
