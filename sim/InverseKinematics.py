"""
Closed-form inverse / forward kinematics for one 3-DOF Mini-SPOT leg.
Fast enough to run inside RL training. Used by the 3D Bezier gait in sim/bezier3d.py.
Pure math: the leg dimensions come in as plain arrays (read from MuJoCo by sim/mujoco_driver.py).
"""

import numpy as np


def _rot_angle(x, z):
    """Angle of a vector in the x-z plane, measured from straight down (-z), positive to -x.
    A rotation about +y by q adds q to this angle (MuJoCo's thigh / knee axis)."""
    return np.arctan2(-x, -z)


class LegIK:
    """Closed-form IK / FK for one 3-DOF Mini-SPOT leg.

    Frames: everything in the trunk (body) frame. The hip (HAA) joint turns about x at the hip
    body origin; the thigh (HFE) and knee (KFE) joints turn about y.

    hip:   hip origin in the trunk frame        thigh: thigh pivot in the hip frame
    calf:  knee in the thigh frame              foot:  foot in the calf frame      (each (x, y, z))
    """

    def __init__(self, hip, thigh, calf, foot):
        self.hip = np.asarray(hip, dtype=float)           # hip origin in the trunk frame
        thigh, calf, foot = (np.asarray(v, dtype=float) for v in (thigh, calf, foot))
        self.tx, self.tz = thigh[0], thigh[2]             # thigh pivot (x, z) in the hip frame
        self.d = thigh[1] + calf[1] + foot[1]             # sideways offset of the foot (constant)
        self.l1 = np.array([calf[0], calf[2]])            # thigh pivot -> knee   (x, z)
        self.l2 = np.array([foot[0], foot[2]])            # knee -> foot          (x, z)
        A = self.l1 @ self.l2
        B = self.l1[0] * self.l2[1] - self.l1[1] * self.l2[0]
        self.R, self.gamma = np.hypot(A, B), np.arctan2(B, A)
        self.ll = self.l1 @ self.l1 + self.l2 @ self.l2

    def fk(self, q):
        """(hip, thigh, knee) -> foot position in the trunk frame."""
        q0, q1, q2 = q
        c2, s2 = np.cos(q2), np.sin(q2)
        u = self.l1 + np.array([c2 * self.l2[0] + s2 * self.l2[1], -s2 * self.l2[0] + c2 * self.l2[1]])
        c1, s1 = np.cos(q1), np.sin(q1)
        x = self.tx + c1 * u[0] + s1 * u[1]
        z = self.tz - s1 * u[0] + c1 * u[1]
        c0, s0 = np.cos(q0), np.sin(q0)
        return self.hip + np.array([x, c0 * self.d - s0 * z, s0 * self.d + c0 * z])

    def ik(self, f):
        """Foot position in the trunk frame -> (hip, thigh, knee) angles, knee bent backward."""
        px, py, pz = np.asarray(f, dtype=float) - self.hip
        # hip (about x): the leg plane sits at sideways offset d; find its height zl
        r_yz2 = py * py + pz * pz
        zl = -np.sqrt(max(r_yz2 - self.d * self.d, 1e-9))
        q0 = np.arctan2(py, -pz) - np.arctan2(self.d, -zl)
        # thigh + knee (about y), target in the thigh-pivot frame
        x, z = px - self.tx, zl - self.tz
        k = np.clip((x * x + z * z - self.ll) / (2.0 * self.R), -1.0, 1.0)
        q2 = self.gamma - np.arccos(k)
        c2, s2 = np.cos(q2), np.sin(q2)
        u = self.l1 + np.array([c2 * self.l2[0] + s2 * self.l2[1], -s2 * self.l2[0] + c2 * self.l2[1]])
        q1 = _rot_angle(x, z) - _rot_angle(*u)
        return np.array([q0, q1, q2])
