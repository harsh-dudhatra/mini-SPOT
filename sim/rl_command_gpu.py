"""GPU training for the command policy (same env / reward / PPO setup as sim/rl_command.py).

Tuned for: NVIDIA RTX 5080 (16 GB), 16-core CPU, 64 GB RAM.

    python -m sim.rl_command_gpu                        # 15 parallel sims, network updates on the GPU
    python -m sim.rl_command_gpu --steps 20000000       # longer run
    python -m sim.rl_command_gpu --envs 12              # fewer sims if the PC is busy
    python -m sim.rl_command_gpu --net 256 256          # bigger network (more GPU work per update)

Where the time goes: MuJoCo physics runs on the CPU (one sim per core), the GPU does the PPO
gradient updates. So the 16 cores set the speed of collecting experience; the GPU makes the
update phase fast and lets you use a bigger network / bigger batches at no extra wall-clock cost.

Saves models/command_gpu_ppo.zip + models/command_gpu_ppo_vecnormalize.pkl, checkpoints every
250 k steps (models/command_gpu_ckpt_*), and on Ctrl+C models/command_gpu_ppo_interrupted.
Evaluate / drive it with the normal script:

    python -m sim.rl_command eval  --model models/command_gpu_ppo
    python -m sim.rl_command drive --model models/command_gpu_ppo
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # avoid torch/MuJoCo thread-pool clashes

import argparse
import time
from pathlib import Path

from sim.DDRandomisationWrapper import make_env
from sim.rl_command import norm_path_for

MODEL_PATH = Path("models") / "command_gpu_ppo"
CPU_CORES = 16
DEFAULT_ENVS = CPU_CORES - 1                         # one sim per core, one core left for the learner


def train(total_steps, n_envs, net_arch, model_path, n_steps, n_epochs):
    # Initialize torch before creating MuJoCo models to avoid thread-pool clashes.
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available: install the GPU build (setup_windows.bat, or "
                         "`uv sync --extra cu128`) and check `nvidia-smi` works.")
    print(f">> GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | {n_envs} sims")
    torch.set_num_threads(2)                         # learner process: CPU side of the update loop
    torch.backends.cuda.matmul.allow_tf32 = True     # Ampere+ (incl. RTX 50xx): faster fp32 matmuls
    torch.backends.cudnn.allow_tf32 = True

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = model_path.name.removesuffix("_ppo") + "_ckpt"
    # "spawn" works on both Windows and Linux and keeps CUDA out of the sim workers
    venv = make_vec_env(make_env, n_envs=n_envs,
                        vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method="spawn"))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=0.99)

    rollout = n_steps * n_envs                       # samples per update
    model = PPO(
        "MlpPolicy", venv, verbose=1, device="cuda",
        n_steps=n_steps, batch_size=rollout // 4, n_epochs=n_epochs,   # 4 minibatches, as in rl_command
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
        policy_kwargs=dict(net_arch=list(net_arch), log_std_init=-1.0),
    )
    checkpoint = CheckpointCallback(save_freq=max(1, 250_000 // n_envs), save_path=str(model_path.parent),
                                    name_prefix=prefix, save_vecnormalize=True)
    t0 = time.time()
    try:
        model.learn(total_timesteps=total_steps, callback=checkpoint)
    except KeyboardInterrupt:
        model_path = model_path.with_name(model_path.name + "_interrupted")
        print(f"\ninterrupted at {model.num_timesteps} steps")
    finally:
        model.save(str(model_path))
        venv.save(str(norm_path_for(model_path)))
        venv.close()
        mins = (time.time() - t0) / 60
        print(f"saved {model_path}.zip and {norm_path_for(model_path)} "
              f"({model.num_timesteps} steps in {mins:.1f} min)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=10_000_000, help="total control steps")
    parser.add_argument("--envs", type=int, default=DEFAULT_ENVS, help="parallel simulations (CPU processes)")
    parser.add_argument("--net", type=int, nargs="+", default=[128, 128], help="hidden layer sizes")
    parser.add_argument("--n-steps", type=int, default=1024, help="steps per sim per rollout")
    parser.add_argument("--epochs", type=int, default=5, help="PPO epochs per rollout")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="where to save")
    args = parser.parse_args()
    train(args.steps, args.envs, args.net, args.model, args.n_steps, args.epochs)


if __name__ == "__main__":                           # required for "spawn" workers on Windows
    main()
