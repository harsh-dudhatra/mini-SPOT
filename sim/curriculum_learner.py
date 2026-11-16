"""Curriculum wrapper for CommandGaitEnv: terrain and push ranges grow with the step count.

    env = CurriculumLearner(CommandGaitEnv(), steps_weight=1e-4)

On every reset, each range below is set to MAX_RANGES * min(1, steps * steps_weight), so it starts
flat with no pushes and reaches full ranges after 1 / steps_weight steps. Steps are counted per env
(per SubprocVecEnv worker).
"""
import gymnasium as gym

# Ranges at full difficulty. Every key is a CommandGaitEnv attribute.
MAX_RANGES = dict(
    bump_height=(0.0, 0.06),      # m
    slope_deg=(-15.0, 15.0),      # deg
    side_slope_deg=(-8.0, 8.0),   # deg
    push_force=(30.0, 100.0),     # N
)


def remap_ranges(step, step_weight):
    """Ranges for this step: MAX_RANGES scaled by min(1, step * step_weight)."""
    scale = min(1.0, step * step_weight)
    return {key: (low * scale, high * scale) for key, (low, high) in MAX_RANGES.items()}


class CurriculumLearner(gym.Wrapper):
    def __init__(self, env, steps_weight=1e-4):
        super().__init__(env)
        self.steps_weight = steps_weight
        self.steps = 0

    def step(self, action):
        self.steps += 1
        return self.env.step(action)

    def reset(self, **kwargs):
        base = self.env.unwrapped
        for key, rng in remap_ranges(self.steps, self.steps_weight).items():
            setattr(base, key, rng)
        if base.hfield < 0:                          # flat scene has no bumps
            base.bump_height = (0.0, 0.0)
        return self.env.reset(**kwargs)
