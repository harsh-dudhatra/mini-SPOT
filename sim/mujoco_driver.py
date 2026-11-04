"""
MuJoCo side of the 3D Bezier gait: everything that reads the MuJoCo model lives here.

  - leg geometry (hip / thigh / calf / foot offsets) -> builds one LegIK per leg
  - actuator indices of (hip, thigh, knee) per leg
  - actuator control ranges

sim/bezier3d.py (gait curves) and sim/InverseKinematics.py (IK / FK) never touch the model;
they only see plain numbers and arrays.
"""

import numpy as np
import mujoco

from .InverseKinematics import LegIK


def _id(model, obj, name):
    return mujoco.mj_name2id(model, obj, name)


def leg_geometry(model, leg):
    """Offsets of one leg from the model: (hip, thigh, calf, foot), each an (x, y, z) array.
    hip in the trunk frame; thigh in the hip frame; calf in the thigh frame; foot in the calf frame."""
    body = lambda name: model.body_pos[_id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()
    foot = model.site_pos[_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot_site")].copy()
    return body(f"{leg}_hip"), body(f"{leg}_thigh"), body(f"{leg}_calf"), foot


class MujocoDriver:
    """Reads the MuJoCo model once; turns per-leg joint angles into an actuator ctrl vector."""

    def __init__(self, model, leg_names):
        self.nu = model.nu
        self.legs = [LegIK(*leg_geometry(model, leg)) for leg in leg_names]
        aid = lambda n: _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self.act = [(aid(f"{l}_hip"), aid(f"{l}_thigh"), aid(f"{l}_calf")) for l in leg_names]
        if min(min(a) for a in self.act) < 0:
            raise ValueError("MujocoDriver needs the 12-actuator model (hip, thigh, calf per leg)")
        self.ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = model.actuator_ctrlrange[:, 1].copy()

    def to_ctrl(self, joint_angles):
        """(4, 3) joint angles (hip, thigh, knee per leg) -> clipped ctrl vector of size model.nu."""
        ctrl = np.zeros(self.nu)
        for ids, q in zip(self.act, joint_angles):
            ctrl[list(ids)] = q
        return np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)

    def joint_targets(self, foot_targets):
        """(4, 3) foot positions in the trunk frame -> IK -> clipped ctrl vector."""
        return self.to_ctrl([leg.ik(f) for leg, f in zip(self.legs, foot_targets)])
