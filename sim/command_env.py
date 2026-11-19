r"""
One RL policy for every movement command, following the D^2-GMBC architecture
(Rahme et al. 2020, arXiv 2010.12070, Fig. 2 / Algorithm 1):

    command (vx, vy, wz) ---------------------------+
                                                    v
    IMU + gait clock + command --> policy --> beta = (psi, delta)  --> 3D Bezier gait (sim/bezier3d.py)
                                         \--> delta_f (12)  ----------------(+)--> IK --> 12 servos

  Commands: vx, vy and wz are each drawn from their own range in CMD_RANGES (forward/backward,
  strafe and turn mixed freely), with a chance of a pure stand. They are re-drawn every few
  seconds, so one network learns all of them and the transitions between them.

  action (14, all in [-1, 1]; 0 = the plain paper gait):
      psi    swing clearance height   0.04 m +- 0.02 m
      delta  stance penetration depth 0.005 m +- 0.005 m
      delta_f: x, y, z foot offset per leg, +- 1.5 cm each (kept small on purpose)
  obs: gravity direction (3) + gyro (3) + gait phase sin/cos (2) + command (3) + last action (14)
       (IMU only, like the paper; no foot contact or joint sensors)

  Terrain, slopes, pushes and dynamics / sensor randomisation live in
  sim/DDRandomisationWrapper.py (use make_env() there to get the wrapped env).
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

# Command ranges, body frame: (low, high) per axis. Each axis is drawn on its own.
CMD_RANGES = {
    "vx": (-0.18, 0.20),     # m/s,   backward (-) / forward (+)
    "vy": (-0.12, 0.12),     # m/s,   strafe right (-) / left (+)
    "wz": (-0.6, 0.6),       # rad/s, turn right (-) / left (+)
}
CMD_LOW = np.array([r[0] for r in CMD_RANGES.values()])
CMD_HIGH = np.array([r[1] for r in CMD_RANGES.values()])

# Old named commands, kept for documentation: command name -> (vx, vy, wz) ranges
# COMMANDS = {
#     "forward":      ((0.05, 0.20), (0.0, 0.0), (0.0, 0.0)),
#     "backward":     ((-0.18, -0.05), (0.0, 0.0), (0.0, 0.0)),
#     "strafe_left":  ((0.0, 0.0), (0.04, 0.12), (0.0, 0.0)),
#     "strafe_right": ((0.0, 0.0), (-0.12, -0.04), (0.0, 0.0)),
#     "turn_left":    ((0.0, 0.0), (0.0, 0.0), (0.2, 0.6)),
#     "turn_right":   ((0.0, 0.0), (0.0, 0.0), (-0.6, -0.2)),
#     "stand":        ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
# }
CMD_SCALE = np.array([0.2, 0.12, 0.6])       # normalises the command in the observation

PSI_SCALE, DELTA_SCALE = 0.02, 0.005         # action +-1 -> curve parameter change [m]
RESIDUAL_LIMIT = 0.015                       # m, max foot offset per axis


class CommandGaitEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        render_mode=None,
        cmd_ranges=None,              # {"vx": (lo, hi), "vy": ..., "wz": ...}; default CMD_RANGES
        stand_prob=0.1,               # chance that a sampled command is a pure stand (0, 0, 0)
        command=None,                 # fix one (vx, vy, wz) command instead of sampling
        command_switch=(3.0, 6.0),    # s between command changes within an episode
        t_swing=0.2,                  # s, paper value
        control_freq=50.0,            # Hz, policy + Bezier update rate
        max_episode_s=10.0,
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
        ranges = dict(CMD_RANGES, **(cmd_ranges or {}))
        self.cmd_ranges = [ranges[k] for k in ("vx", "vy", "wz")]
        self.stand_prob = stand_prob
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

        self.w = dict(lin=1.0, yaw=0.5, upright=1.0, height=1.0, mod=0.05,
                      smooth=0.1, alive=0.5, fall=10.0)
        if reward_weights:
            self.w.update(reward_weights)

        self.act_dim = 2 + 12
        self.action_space = spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (3 + 3 + 2 + 3 + self.act_dim,), np.float32)
        self.rng = np.random.default_rng(seed)
        self.latency = 0

    # ------------------------------------------------------------ commands
    def set_command(self, cmd, name="custom"):
        """Change the command on the fly (e.g. from the keyboard)."""
        self.cmd = np.asarray(cmd, dtype=float)
        self.cmd_name = name

    def _sample_command(self):
        if self.fixed_command is not None:
            self.set_command(self.fixed_command, "fixed")
        elif self.rng.uniform() < self.stand_prob:
            self.set_command(np.zeros(3), "stand")
        else:
            self.set_command([self.rng.uniform(*r) for r in self.cmd_ranges], "sampled")
        self.next_switch = self.data.time + self.rng.uniform(*self.command_switch)

    # ------------------------------------------------------------ terrain
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

    # ------------------------------------------------------------ sensing
    def _obs(self):
        R = self.data.xmat[self.trunk].reshape(3, 3)
        g = self.model.opt.gravity
        grav = R.T @ (g / np.linalg.norm(g))       # what the IMU feels as "down"
        gyro = self.data.qvel[self.root_dof + 3:self.root_dof + 6].copy()
        obs = np.concatenate([grav, gyro,
                              [np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)],
                              self.cmd / CMD_SCALE, self.prev_action])
        return obs.astype(np.float32)

    def _body_height(self):
        x, y, z = self.data.qpos[self.root_qpos:self.root_qpos + 3]
        return float(z - self._ground_height(x, y))

    def _body_state(self):
        """Trunk state in the body frame, used by the reward, termination and info."""
        R = self.data.xmat[self.trunk].reshape(3, 3)
        return dict(
            up=R.T @ np.array([0.0, 0.0, 1.0]),    # ground normal (world z) in the body frame
            v=R.T @ self.data.qvel[self.root_dof:self.root_dof + 3],
            w=self.data.qvel[self.root_dof + 3:self.root_dof + 6].copy(),
            h=self._body_height(),
        )

    # ------------------------------------------------------------ reward: follow the command
    def _reward(self, action, s, terminated):
        """Returns (reward, terms). s = _body_state() after the step."""
        terms = {
            "lin": self.w["lin"] * np.exp(-float(np.sum((s["v"][:2] - self.cmd[:2]) ** 2)) / 0.005),
            "yaw": self.w["yaw"] * np.exp(-float((s["w"][2] - self.cmd[2]) ** 2) / 0.05),
            "upright": -self.w["upright"] * float(np.sum(s["up"][:2] ** 2)),
            "height": -self.w["height"] * abs(s["h"] - self.target_height) / self.target_height,
            "mod": -self.w["mod"] * float(np.mean(action ** 2)),          # keep changes small
            "smooth": -self.w["smooth"] * float(np.mean((action - self.prev_action) ** 2)),
            "alive": self.w["alive"],
        }
        reward = float(sum(terms.values()))
        if terminated:
            reward -= self.w["fall"]
        return reward, terms

    def _terminated(self, s):
        return bool(s["up"][2] < 0.5 or s["h"] < 0.5 * self.target_height
                    or not np.all(np.isfinite(self.data.qpos)))

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        """options: phase (gait start phase, default 0), latency (0/1 control steps, default 0)."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        mujoco.mj_resetData(self.model, self.data)

        self.cmd = np.zeros(3)
        self.phase = float(options.get("phase", 0.0))
        self.latency = int(options.get("latency", 0))
        self.prev_action = np.zeros(self.act_dim)
        self.pending_action = np.zeros(self.act_dim)
        self.step_count = 0

        # stand, settle, then start the gait
        self.data.ctrl[:] = self.driver.joint_targets(self.gen.f_stand)
        for _ in range(int(0.5 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self.target_height = self._body_height()
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
            mujoco.mj_step(self.model, self.data)

        s = self._body_state()
        terminated = self._terminated(s)
        reward, terms = self._reward(action, s, terminated)
        self.prev_action = action
        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        info = {"reward_terms": terms, "command": self.cmd_name, "cmd": self.cmd.copy(),
                "vel": np.array([s["v"][0], s["v"][1], s["w"][2]])}
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
