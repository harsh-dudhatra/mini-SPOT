"""Train / evaluate / drive the ONE policy for all movement commands (paper architecture).

    python -m sim.rl_command train                     # PPO, 8 parallel sims, saves models/command_ppo.zip
    python -m sim.rl_command train --steps 1000000     # shorter run
    python -m sim.rl_command eval                      # plain paper gait vs gait + policy, per command
    python -m sim.rl_command eval --scenario mixed     # same on bumps + slopes
    python -m sim.rl_command drive                     # 3D viewer, type commands (macOS: use mjpython)
    python -m sim.rl_command drive --plain             # same with the plain paper gait (no network needed)
    python -m sim.rl_command drive --scenario bumps6   # drive on bumpy ground

Drive commands (type + Enter):  w forward   s backward   a strafe left   d strafe right
                                q turn left   e turn right   x stop   (or the full names)

Ctrl+C during training saves to models/command_ppo_interrupted; checkpoints every 250 k steps
(models/command_ckpt_*) can be loaded with --model.
See sim/command_env.py (environment) and sim/bezier3d.py (3D Bezier gait + IK).
"""
import argparse
import select
import sys
import time
from pathlib import Path

import numpy as np

from sim.command_env import COMMANDS, CommandGaitEnv

MODEL_PATH = Path("models") / "command_ppo"

_OFF = dict(bump_height=(0.0, 0.0), slope_deg=(0.0, 0.0), side_slope_deg=(0.0, 0.0), push_force=(0.0, 0.0))
SCENARIOS = {                     # fixed test conditions (robot dynamics still randomized)
    "flat":      dict(_OFF),
    "bumps4":    dict(_OFF, bump_height=(0.04, 0.04)),
    "bumps6":    dict(_OFF, bump_height=(0.06, 0.06)),
    "uphill10":  dict(_OFF, slope_deg=(10.0, 10.0)),
    "uphill15":  dict(_OFF, slope_deg=(15.0, 15.0)),
    "downhill15": dict(_OFF, slope_deg=(-15.0, -15.0)),
    "side8":     dict(_OFF, side_slope_deg=(8.0, 8.0)),
    "push80":    dict(_OFF, push_force=(80.0, 80.0)),
    "mixed":     dict(),          # the training distribution: everything at once
}

# typical command used for eval / drive: the middle of each training range
TYPICAL = {name: tuple(float(np.mean(r)) for r in rng) for name, rng in COMMANDS.items()}
KEYS = {"w": "forward", "s": "backward", "a": "strafe_left", "d": "strafe_right",
        "q": "turn_left", "e": "turn_right", "x": "stand", "stop": "stand",
        "left": "strafe_left", "right": "strafe_right",
        "turn left": "turn_left", "turn right": "turn_right"}


def train(total_steps, n_envs):
    train_ppo(CommandGaitEnv, total_steps, n_envs, MODEL_PATH)


def train_ppo(env_factory, total_steps, n_envs, model_path):
    """PPO with parallel sims. Saves <model_path>.zip + its normalisation file, a checkpoint every
    250 k steps (<name>_ckpt_*), and on Ctrl+C <model_path>_interrupted instead of model_path."""
    # Initialize torch before creating MuJoCo models to avoid thread-pool clashes.
    import torch

    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import CheckpointCallback

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = model_path.name.removesuffix("_ppo") + "_ckpt"
    venv = make_vec_env(env_factory, n_envs=n_envs,
                        vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method="spawn"))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=0.99)

    model = PPO(
        "MlpPolicy", venv, verbose=1,
        n_steps=1024, batch_size=min(4096, 1024 * n_envs), n_epochs=5,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
        policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.0),   # start with small corrections
    )
    checkpoint = CheckpointCallback(save_freq=max(1, 250_000 // n_envs), save_path=str(model_path.parent),
                                    name_prefix=prefix, save_vecnormalize=True)
    try:
        model.learn(total_timesteps=total_steps, callback=checkpoint)
    except KeyboardInterrupt:
        model_path = model_path.with_name(model_path.name + "_interrupted")
        print(f"\ninterrupted at {model.num_timesteps} steps")
    finally:
        model.save(str(model_path))
        venv.save(str(norm_path_for(model_path)))
        venv.close()
        print(f"saved {model_path}.zip and {norm_path_for(model_path)}")


def norm_path_for(model_path):
    """The observation-normalisation file that belongs to a model / checkpoint."""
    model_path = Path(str(model_path).removesuffix(".zip"))
    if "_ckpt_" in model_path.name:                      # SB3 checkpoint naming
        prefix, steps = model_path.name.split("_ckpt_", 1)
        return model_path.with_name(f"{prefix}_ckpt_vecnormalize_{steps}.pkl")
    return model_path.with_name(model_path.name + "_vecnormalize.pkl")


def load_policy(model_path=MODEL_PATH):
    """Returns act(obs) -> action, using the saved obs normalisation."""
    import pickle
    from stable_baselines3 import PPO

    model_path = Path(str(model_path).removesuffix(".zip"))
    norm_path = norm_path_for(model_path)
    if not norm_path.exists():
        raise FileNotFoundError(f"{norm_path} is missing: it holds the input normalisation for "
                                f"{model_path}.zip (checkpoints from before this fix don't have one)")
    model = PPO.load(str(model_path), device="cpu")
    with open(norm_path, "rb") as f:
        norm = pickle.load(f)
    rms, clip, eps = norm.obs_rms, norm.clip_obs, norm.epsilon

    def act(obs):
        o = np.clip((obs - rms.mean) / np.sqrt(rms.var + eps), -clip, clip)
        return model.predict(o.astype(np.float32), deterministic=True)[0]
    return act


def plain_policy():
    return lambda obs: np.zeros(14)                  # action 0 = plain paper gait


def run(act, command, scenario, episodes):
    """Returns (episodes survived, mean achieved (vx, vy, wz) after the first second)."""
    alive, vels = 0, []
    for ep in range(episodes):
        env = CommandGaitEnv(command=TYPICAL[command], **SCENARIOS[scenario])
        obs, _ = env.reset(seed=2000 + ep)
        t0 = env.data.time
        while True:
            obs, _, term, trunc, info = env.step(act(obs))
            if env.data.time - t0 > 1.0:
                vels.append(info["vel"])
            if term or trunc:
                break
        alive += not term
        env.close()
    return alive, np.mean(vels, axis=0)


def evaluate(scenario, episodes, model_path, commands):
    act = load_policy(model_path)
    print(f"scenario '{scenario}', {episodes} episodes of 10 s per command; "
          f"achieved (vx m/s, vy m/s, wz rad/s)")
    print(f"{'command':>13} {'asked':>20} | {'plain gait':>30} | {'gait + policy':>30}")
    fmt = lambda v: f"({v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f})"
    for name in commands:
        base = run(plain_policy(), name, scenario, episodes)
        pol = run(act, name, scenario, episodes)
        print(f"{name:>13} {fmt(TYPICAL[name]):>20} | {base[0]:>2}/{episodes} up {fmt(base[1]):>20} | "
              f"{pol[0]:>2}/{episodes} up {fmt(pol[1]):>20}")


def drive(scenario, model_path, plain):
    env = CommandGaitEnv(render_mode="human", command=TYPICAL["stand"],
                         **dict(SCENARIOS[scenario], max_episode_s=1e9))
    act = plain_policy() if plain else load_policy(model_path)
    obs, _ = env.reset()
    print(__doc__.split("Drive commands")[1].split("Ctrl+C")[0].strip())
    print(">> standing - type a command")
    while True:
        t0 = time.time()
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip().lower().replace("_", " ")
            name = KEYS.get(line, line.replace(" ", "_"))
            if name in TYPICAL:
                env.set_command(TYPICAL[name], name)
                print(f">> {name}  (vx, vy, wz) = {TYPICAL[name]}")
            elif line:
                print(">> unknown command; use w a s d q e x")
        obs, _, term, trunc, info = env.step(act(obs))
        if term:
            print(">> fell - resetting")
            obs, _ = env.reset()
            env.set_command(TYPICAL["stand"], "stand")
        time.sleep(max(0.0, env.control_dt - (time.time() - t0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("train", "eval", "drive"))
    parser.add_argument("--steps", type=int, default=5_000_000, help="train: total control steps")
    parser.add_argument("--envs", type=int, default=7, help="train: parallel simulations")
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="flat",
                        help="eval/drive: terrain (training always uses the full random mix)")
    parser.add_argument("--commands", nargs="+", choices=list(COMMANDS), default=list(COMMANDS),
                        help="eval: which commands")
    parser.add_argument("--episodes", type=int, default=6, help="eval: episodes per command")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="eval/drive: model or checkpoint")
    parser.add_argument("--plain", action="store_true", help="drive: plain paper gait, no network")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.steps, args.envs)
    elif args.mode == "eval":
        evaluate(args.scenario, args.episodes, args.model, args.commands)
    else:
        drive(args.scenario, args.model, args.plain)


if __name__ == "__main__":
    main()
