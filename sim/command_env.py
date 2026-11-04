"""
One RL policy for every movement command, following the D^2-GMBC architecture
(Rahme et al. 2020, arXiv 2010.12070, Fig. 2 / Algorithm 1):

    command (vx, vy, wz) ---------------------------+
                                                    v
    IMU + gait clock + command --> policy --> beta = (psi, delta)  --> 3D Bezier gait (sim/bezier3d.py)
                                         \--> delta_f (12)  ----------------(+)--> IK --> 12 servos

  Commands: forward, backward, strafe left / right, turn left / right, stand. They are sampled
  every episode and switched every few seconds, so one network learns all of them and the
  transitions between them.

  action (14, all in [-1, 1]; 0 = the plain paper gait):
      psi    swing clearance height   0.04 m +- 0.02 m
      delta  stance penetration depth 0.005 m +- 0.005 m
      delta_f: x, y, z foot offset per leg, +- 1.5 cm each (kept small on purpose)
  obs: gravity direction (3) + gyro (3) + gait phase sin/cos (2) + command (3) + last action (14)
       (IMU only, like the paper; no foot contact or joint sensors)

  Training conditions, re-drawn every episode:
      terrain     random bumps 0-6 cm (assets/scene_terrain.xml heightfield)
      slopes      up/downhill +-15 deg, sideways +-8 deg (gravity is tilted: same physics)
      pushes      off by default (push_force=(30, 100) to add them)
      dynamics    friction, trunk mass, motor strength (kp), sensor noise, 0-1 step latency
"""

import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from .bezier3d import DELTA_DEFAULT, LEG_NAMES, PSI_DEFAULT, Bezier3DGait
from .mujoco_driver import MujocoDriver

HERE = os.path.dirname(os.path.abspath(__file__))
FLAT_XML = os.path.join(HERE, "..", "assets", "scene.xml")
TERRAIN_XML = os.path.join(HERE, "..", "assets", "scene_terrain.xml")

# command name -> (vx, vy, wz) ranges, body frame [m/s, m/s, rad/s]
COMMANDS = {
    "forward":      ((0.05, 0.20), (0.0, 0.0), (0.0, 0.0)),
    "backward":     ((-0.18, -0.05), (0.0, 0.0), (0.0, 0.0)),
    "strafe_left":  ((0.0, 0.0), (0.04, 0.12), (0.0, 0.0)),
    "strafe_right": ((0.0, 0.0), (-0.12, -0.04), (0.0, 0.0)),
    "turn_left":    ((0.0, 0.0), (0.0, 0.0), (0.2, 0.6)),
    "turn_right":   ((0.0, 0.0), (0.0, 0.0), (-0.6, -0.2)),
    "stand":        ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
}
CMD_SCALE = np.array([0.2, 0.12, 0.6])       # normalises the command in the observation

PSI_SCALE, DELTA_SCALE = 0.02, 0.005         # action +-1 -> curve parameter change [m]
RESIDUAL_LIMIT = 0.015                       # m, max foot offset per axis


def _smooth_noise(rng, n, cells):
    """n x n smooth random field in [0, 1]: random grid of `cells` points, bilinear upsampled."""
    coarse = rng.uniform(0.0, 1.0, (cells, cells))
    x = np.linspace(0, cells - 1, n)
    rows = np.array([np.interp(x, np.arange(cells), r) for r in coarse])           # (cells, n)
    return np.array([np.interp(x, np.arange(cells), c) for c in rows.T]).T        # (n, n)


class CommandGaitEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        commands=tuple(COMMANDS),     # which commands to train / test on
        command=None,                 # fix one (vx, vy, wz) command instead of sampling
        command_switch=(3.0, 6.0),    # s between command changes within an episode
        t_swing=0.2,                  # s, paper value
        control_freq=50.0,            # Hz, policy + Bezier update rate
        max_episode_s=10.0,
        # training conditions; each is a (low, high) range drawn per episode. (0, 0) = off
        bump_height=(0.0, 0.06),      # m, terrain bumps (needs the terrain scene)
        slope_deg=(-15.0, 15.0),      # uphill (+) / downhill (-) along the walking direction
        side_slope_deg=(-8.0, 8.0),
        push_force=(0.0, 0.0),        # N, horizontal push; off by default, e.g. (30, 100) to train with pushes
        push_duration=0.15,           # s
        push_interval=(1.5, 3.0),     # s between pushes
        randomize=True,               # friction / mass / motor strength / noise / latency
        terrain=True,                 # False: flat floor scene (faster, no bumps)
        reward_weights=None,
        seed=None,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(TERRAIN_XML if terrain else FLAT_XML)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None
        m = self.model

        self.driver = MujocoDriver(m, LEG_NAMES)
        self.gen = Bezier3DGait(self.driver.legs, t_swing=t_swing)
        self.commands = list(commands)
        self.fixed_command = None if command is None else np.asarray(command, dtype=float)
        self.command_switch = command_switch
        self.cmd = np.zeros(3)
        self.cmd_name = "stand"

        self.control_dt = 1.0 / control_freq
        self.n_sim = max(1, round(self.control_dt / m.opt.timestep))
        self.max_steps = int(max_episode_s / self.control_dt)

        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
        self.root_qpos = m.jnt_qposadr[free]
        self.root_dof = m.jnt_dofadr[free]
        self.trunk = m.jnt_bodyid[free]

        self.hfield = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        if self.hfield >= 0:
            self.terrain_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
            self.hf_nrow, self.hf_ncol = m.hfield_nrow[self.hfield], m.hfield_ncol[self.hfield]
            self.hf_size = m.hfield_size[self.hfield].copy()     # radius x, radius y, height, base

        # nominal values for domain randomization
        self.nominal_friction = m.geom_friction.copy()
        self.nominal_mass = m.body_mass.copy()
        self.nominal_inertia = m.body_inertia.copy()
        self.nominal_gain = m.actuator_gainprm.copy()
        self.nominal_bias = m.actuator_biasprm.copy()

        self.randomize = randomize
        self.bump_height = bump_height if self.hfield >= 0 else (0.0, 0.0)
        self.slope_deg = slope_deg
        self.side_slope_deg = side_slope_deg
        self.push_force = push_force
        self.push_duration = push_duration
        self.push_interval = push_interval

        self.w = dict(lin=1.0, yaw=0.5, upright=1.0, height=1.0, mod=0.05,
                      smooth=0.1, alive=0.5, fall=10.0)
        if reward_weights:
            self.w.update(reward_weights)

        self.act_dim = 2 + 12
        self.action_space = spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (3 + 3 + 2 + 3 + self.act_dim,), np.float32)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ commands
    def set_command(self, cmd, name="custom"):
        """Change the command on the fly (e.g. from the keyboard)."""
        self.cmd = np.asarray(cmd, dtype=float)
        self.cmd_name = name

    def _sample_command(self):
        if self.fixed_command is not None:
            self.set_command(self.fixed_command, "fixed")
        else:
            name = self.commands[self.rng.integers(len(self.commands))]
            self.set_command([self.rng.uniform(*r) for r in COMMANDS[name]], name)
        self.next_switch = self.data.time + self.rng.uniform(*self.command_switch)

    # ------------------------------------------------------------ terrain
    def _make_terrain(self):
        if self.hfield < 0:
            return
        amp = self.rng.uniform(*self.bump_height)
        n = self.hf_nrow
        cells = int(self.rng.integers(20, 50))                 # bump size ~16-40 cm
        field = _smooth_noise(self.rng, n, cells) * amp / self.hf_size[2]
        self.model.hfield_data[:] = np.clip(field, 0.0, 1.0).ravel()
        # put the ground under the start position at z = 0
        self.model.geom_pos[self.terrain_geom, 2] = 0.0
        self.model.geom_pos[self.terrain_geom, 2] = -self._ground_height(0.0, 0.0)
        if self.viewer is not None:
            self.viewer.update_hfield(self.hfield)

    def _ground_height(self, x, y):
        """Terrain height below (x, y) in world coordinates (0 on the flat scene)."""
        if self.hfield < 0:
            return 0.0
        rx, ry, h = self.hf_size[:3]
        c = int(round((x + rx) / (2 * rx) * (self.hf_ncol - 1)))
        r = int(round((y + ry) / (2 * ry) * (self.hf_nrow - 1)))
        c, r = min(max(c, 0), self.hf_ncol - 1), min(max(r, 0), self.hf_nrow - 1)
        return float(self.model.geom_pos[self.terrain_geom, 2]
                     + self.model.hfield_data[r * self.hf_ncol + c] * h)

    def _set_slope(self):
        """Tilt gravity: + pitch = uphill when walking forward (+x)."""
        th = np.radians(self.rng.uniform(*self.slope_deg))
        ph = np.radians(self.rng.uniform(*self.side_slope_deg))
        self.model.opt.gravity[:] = 9.81 * np.array(
            [-np.sin(th), np.cos(th) * np.sin(ph), -np.cos(th) * np.cos(ph)])
        self.slope = (float(np.degrees(th)), float(np.degrees(ph)))

    # ------------------------------------------------------------ sensing
    def _obs(self):
        R = self.data.xmat[self.trunk].reshape(3, 3)
        g = self.model.opt.gravity
        grav = R.T @ (g / np.linalg.norm(g))       # what the IMU feels as "down"
        gyro = self.data.qvel[self.root_dof + 3:self.root_dof + 6].copy()
        if self.randomize:
            grav = grav + self.rng.normal(0, 0.02, 3)
            gyro = gyro + self.rng.normal(0, 0.05, 3)
        obs = np.concatenate([grav, gyro,
                              [np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)],
                              self.cmd / CMD_SCALE, self.prev_action])
        return obs.astype(np.float32)

    def _body_height(self):
        x, y, z = self.data.qpos[self.root_qpos:self.root_qpos + 3]
        return float(z - self._ground_height(x, y))

    # ------------------------------------------------------------ disturbances
    def _randomize_model(self):
        m, r = self.model, self.rng
        m.geom_friction[:] = self.nominal_friction
        m.body_mass[:] = self.nominal_mass
        m.body_inertia[:] = self.nominal_inertia
        m.actuator_gainprm[:] = self.nominal_gain
        m.actuator_biasprm[:] = self.nominal_bias
        if not self.randomize:
            return
        m.geom_friction[:, 0] *= r.uniform(0.5, 1.25)   # MuJoCo uses the max of both surfaces
        k = r.uniform(0.8, 1.25)                     # trunk mass / payload
        m.body_mass[self.trunk] *= k
        m.body_inertia[self.trunk] *= k
        s = r.uniform(0.8, 1.2)                      # motor strength: kp of the position servos
        m.actuator_gainprm[:, 0] *= s
        m.actuator_biasprm[:, 1] *= s

    def _schedule_push(self):
        lo, hi = self.push_interval
        self.next_push = self.data.time + self.rng.uniform(lo, hi)

    def _apply_push(self):
        d = self.data
        d.xfrc_applied[self.trunk, :] = 0.0
        if self.push_force[1] <= 0:
            return
        if self.push_end is not None and d.time < self.push_end:
            d.xfrc_applied[self.trunk, :3] = self.push_vec
            return
        self.push_end = None
        if d.time >= self.next_push:
            a = self.rng.uniform(0, 2 * np.pi)
            f = self.rng.uniform(*self.push_force)
            self.push_vec = np.array([f * np.cos(a), f * np.sin(a), 0.0])
            self.push_end = d.time + self.push_duration
            d.xfrc_applied[self.trunk, :3] = self.push_vec
            self.n_pushes += 1
            self._schedule_push()

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._randomize_model()
        self._make_terrain()
        self._set_slope()
        mujoco.mj_resetData(self.model, self.data)

        self.cmd = np.zeros(3)
        self.phase = float(self.rng.uniform(0, 1)) if self.randomize else 0.0
        self.prev_action = np.zeros(self.act_dim)
        self.pending_action = np.zeros(self.act_dim)
        self.latency = int(self.rng.integers(0, 2)) if self.randomize else 0
        self.step_count = 0
        self.push_end = None
        self.n_pushes = 0

        # stand, settle, then start the gait
        self.data.ctrl[:] = self.driver.joint_targets(self.gen.f_stand)
        for _ in range(int(0.5 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self.target_height = self._body_height()
        self._schedule_push()
        self.next_push = max(self.next_push, self.data.time + 1.0)   # first second: no push
        self._sample_command()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        applied = self.pending_action if self.latency else action     # servo latency
        self.pending_action = action

        if self.fixed_command is None and self.data.time >= self.next_switch:
            self._sample_command()
        self.phase = (self.phase + self.control_dt / self.gen.t_stride) % 1.0
        psi = PSI_DEFAULT + PSI_SCALE * applied[0]
        delta = DELTA_DEFAULT + DELTA_SCALE * applied[1]
        residual = applied[2:].reshape(4, 3) * RESIDUAL_LIMIT
        feet = self.gen.foot_targets(self.phase, self.cmd, psi, delta, residual)
        self.data.ctrl[:] = self.driver.joint_targets(feet)
        for _ in range(self.n_sim):
            self._apply_push()
            mujoco.mj_step(self.model, self.data)

        # ------------------------------------------------ reward: follow the command
        R = self.data.xmat[self.trunk].reshape(3, 3)
        up_body = R.T @ np.array([0.0, 0.0, 1.0])     # ground normal (world z) in the body frame
        v_body = R.T @ self.data.qvel[self.root_dof:self.root_dof + 3]
        w_body = self.data.qvel[self.root_dof + 3:self.root_dof + 6]
        h = self._body_height()
        terms = {
            "lin": self.w["lin"] * np.exp(-float(np.sum((v_body[:2] - self.cmd[:2]) ** 2)) / 0.005),
            "yaw": self.w["yaw"] * np.exp(-float((w_body[2] - self.cmd[2]) ** 2) / 0.05),
            "upright": -self.w["upright"] * float(np.sum(up_body[:2] ** 2)),
            "height": -self.w["height"] * abs(h - self.target_height) / self.target_height,
            "mod": -self.w["mod"] * float(np.mean(action ** 2)),          # keep changes small
            "smooth": -self.w["smooth"] * float(np.mean((action - self.prev_action) ** 2)),
            "alive": self.w["alive"],
        }
        reward = float(sum(terms.values()))
        self.prev_action = action
        self.step_count += 1

        terminated = bool(up_body[2] < 0.5 or h < 0.5 * self.target_height
                          or not np.all(np.isfinite(self.data.qpos)))
        if terminated:
            reward -= self.w["fall"]
        truncated = self.step_count >= self.max_steps
        info = {"reward_terms": terms, "command": self.cmd_name, "cmd": self.cmd.copy(),
                "vel": np.array([v_body[0], v_body[1], w_body[2]]), "slope": self.slope,
                "pushes": self.n_pushes}
        if self.render_mode == "human":
            self.render()
        return np.nan_to_num(self._obs()), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
