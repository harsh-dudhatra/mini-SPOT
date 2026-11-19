"""Domain + dynamics randomisation for CommandGaitEnv, as a gym wrapper.

    env = DDRandomisationWrapper(CommandGaitEnv())          # or make_env(**kwargs)

Re-drawn every episode (on reset):
    domain      terrain bumps 0-6 cm (needs the terrain scene, assets/scene_terrain.xml heightfield)
                slopes uphill/downhill +-15 deg, sideways +-8 deg (gravity is tilted: same physics)
    dynamics    friction, trunk mass / inertia, motor strength (kp)       (randomize=True)
                0-1 control step action latency, random start phase      (randomize=True)
During the episode:
    pushes      horizontal pushes on the trunk (push_force=(0, 0) = off)
    sensors     gaussian noise on the IMU part of the observation       (randomize=True)

Every range is a (low, high) tuple; (0, 0) switches it off. The ranges are plain attributes, so a
curriculum (sim/curriculum_learner.py) can change them between episodes.
"""
import gymnasium as gym
import numpy as np

from .command_env import CommandGaitEnv

GRAV_NOISE, GYRO_NOISE = 0.02, 0.05      # std of the IMU noise (gravity direction, rad/s)

# constructor keywords that belong to this wrapper (make_env sends the rest to CommandGaitEnv)
DR_KEYS = ("bump_height", "slope_deg", "side_slope_deg", "push_force", "push_duration",
           "push_interval", "randomize")


def _smooth_noise(rng, n, cells):
    """n x n smooth random field in [0, 1]: random grid of `cells` points, bilinear upsampled."""
    coarse = rng.uniform(0.0, 1.0, (cells, cells))
    x = np.linspace(0, cells - 1, n)
    rows = np.array([np.interp(x, np.arange(cells), r) for r in coarse])           # (cells, n)
    return np.array([np.interp(x, np.arange(cells), c) for c in rows.T]).T        # (n, n)


class DDRandomisationWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        bump_height=(0.0, 0.06),      # m, terrain bumps (needs the terrain scene)
        slope_deg=(-15.0, 15.0),      # uphill (+) / downhill (-) along the walking direction
        side_slope_deg=(-8.0, 8.0),
        push_force=(5, 20),           # N, horizontal push; (0, 0) = off, e.g. (30, 100) for hard pushes
        push_duration=0.15,           # s
        push_interval=(1.5, 3.0),     # s between pushes
        randomize=True,               # friction / mass / motor strength / noise / latency / phase
        seed=None,
    ):
        super().__init__(env)
        base = env.unwrapped
        m = base.model
        self.bump_height = bump_height
        self.slope_deg = slope_deg
        self.side_slope_deg = side_slope_deg
        self.push_force = push_force
        self.push_duration = push_duration
        self.push_interval = push_interval
        self.randomize = randomize
        self.rng = np.random.default_rng(seed)

        # nominal values, restored before every draw
        self.nominal_friction = m.geom_friction.copy()
        self.nominal_mass = m.body_mass.copy()
        self.nominal_inertia = m.body_inertia.copy()
        self.nominal_gain = m.actuator_gainprm.copy()
        self.nominal_bias = m.actuator_biasprm.copy()

        self.slope = (0.0, 0.0)
        self.push_end = None
        self.next_push = 0.0
        self.n_pushes = 0

    # ------------------------------------------------------------ domain: terrain + slope
    def _make_terrain(self):
        base = self.env.unwrapped
        if base.hfield < 0:                                    # flat scene: nothing to build
            return
        m = base.model
        amp = self.rng.uniform(*self.bump_height)
        n = base.hf_nrow
        cells = int(self.rng.integers(20, 50))                 # bump size ~16-40 cm
        field = _smooth_noise(self.rng, n, cells) * amp / base.hf_size[2]
        m.hfield_data[:] = np.clip(field, 0.0, 1.0).ravel()
        # put the ground under the start position at z = 0
        m.geom_pos[base.terrain_geom, 2] = 0.0
        m.geom_pos[base.terrain_geom, 2] = -base._ground_height(0.0, 0.0)
        if base.viewer is not None:
            base.viewer.update_hfield(base.hfield)

    def _set_slope(self):
        """Tilt gravity: + pitch = uphill when walking forward (+x)."""
        th = np.radians(self.rng.uniform(*self.slope_deg))
        ph = np.radians(self.rng.uniform(*self.side_slope_deg))
        self.env.unwrapped.model.opt.gravity[:] = 9.81 * np.array(
            [-np.sin(th), np.cos(th) * np.sin(ph), -np.cos(th) * np.cos(ph)])
        self.slope = (float(np.degrees(th)), float(np.degrees(ph)))

    # ------------------------------------------------------------ dynamics
    def _randomize_model(self):
        base = self.env.unwrapped
        m, r = base.model, self.rng
        m.geom_friction[:] = self.nominal_friction
        m.body_mass[:] = self.nominal_mass
        m.body_inertia[:] = self.nominal_inertia
        m.actuator_gainprm[:] = self.nominal_gain
        m.actuator_biasprm[:] = self.nominal_bias
        if not self.randomize:
            return
        m.geom_friction[:, 0] *= r.uniform(0.5, 1.25)   # MuJoCo uses the max of both surfaces
        k = r.uniform(0.8, 1.25)                     # trunk mass / payload
        m.body_mass[base.trunk] *= k
        m.body_inertia[base.trunk] *= k
        s = r.uniform(0.8, 1.2)                      # motor strength: kp of the position servos
        m.actuator_gainprm[:, 0] *= s
        m.actuator_biasprm[:, 1] *= s

    def _noisy(self, obs):
        """IMU noise on obs[0:3] (gravity direction) and obs[3:6] (gyro)."""
        if not self.randomize:
            return obs
        obs = obs.copy()
        obs[0:3] += self.rng.normal(0, GRAV_NOISE, 3)
        obs[3:6] += self.rng.normal(0, GYRO_NOISE, 3)
        return obs.astype(np.float32)

    # ------------------------------------------------------------ pushes
    def _schedule_push(self):
        lo, hi = self.push_interval
        self.next_push = self.env.unwrapped.data.time + self.rng.uniform(lo, hi)

    def _apply_push(self):
        """Set the push force for the next control step (it is held for all its physics steps)."""
        base = self.env.unwrapped
        d = base.data
        d.xfrc_applied[base.trunk, :] = 0.0
        if self.push_force[1] <= 0:
            return
        if self.push_end is not None and d.time < self.push_end:
            d.xfrc_applied[base.trunk, :3] = self.push_vec
            return
        self.push_end = None
        if d.time >= self.next_push:
            a = self.rng.uniform(0, 2 * np.pi)
            f = self.rng.uniform(*self.push_force)
            self.push_vec = np.array([f * np.cos(a), f * np.sin(a), 0.0])
            self.push_end = d.time + self.push_duration
            d.xfrc_applied[base.trunk, :3] = self.push_vec
            self.n_pushes += 1
            self._schedule_push()

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._randomize_model()
        self._make_terrain()
        self._set_slope()
        options = dict(options or {})
        if self.randomize:
            options.setdefault("phase", float(self.rng.uniform(0, 1)))
            options.setdefault("latency", int(self.rng.integers(0, 2)))

        obs, info = self.env.reset(seed=seed, options=options)    # stands + settles on the new world

        self.push_end = None
        self.n_pushes = 0
        self._schedule_push()
        self.next_push = max(self.next_push, self.env.unwrapped.data.time + 1.0)   # first second: no push
        return self._noisy(obs), info

    def step(self, action):
        self._apply_push()
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.update(slope=self.slope, pushes=self.n_pushes)
        return self._noisy(obs), reward, terminated, truncated, info


def make_env(**kwargs):
    """CommandGaitEnv wrapped in DDRandomisationWrapper. Takes the keywords of both
    (DR_KEYS go to the wrapper, everything else to the env); `seed` goes to both."""
    dr = {k: kwargs.pop(k) for k in DR_KEYS if k in kwargs}
    return DDRandomisationWrapper(CommandGaitEnv(**kwargs), seed=kwargs.get("seed"), **dr)
