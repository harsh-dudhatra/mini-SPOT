-# Mini-SPOT

A small 4-legged robot dog (inspired by Boston Dynamics Spot), built as a
Bachelor's project in Robotics Engineering.

Right now everything runs in a **physics simulation** (MuJoCo) on the computer.
The robot walks forward / backward, steps sideways and turns on command, using a 3D Bézier
gait. Reinforcement learning (RL) is being added to keep that working on bumps, slopes and pushes.

> This README describes the `rl-gait` branch.

---

## The 30-second version

- The robot has **4 legs**, each with **3 motors**: a **sideways hip**, a **shoulder** (swings the
  leg forward/back) and a **knee** (bends the leg). 4 × 3 = **12 motors**.
- Walking = moving each foot in a repeating loop: **lift it, swing it, put it down, push back**.
  The loop is a Bézier curve in 3D, so the same gait can go forward, backward, sideways or turn.
  An exact leg IK turns each foot position into the 3 motor angles.
- **RL** = a small neural network watches the IMU 50× per second and slightly reshapes the curve
  and the foot positions, so the robot copes with bumps, slopes and pushes.

---

## What each file does

### The files you actually run

| File | What it is | When to use it |
|---|---|---|
| `sim/rl_command.py` | **One network for every move (paper architecture).** Train / test / **drive** it: forward, backward, strafe, turn. `drive --plain` drives the gait by hand, no network needed | The main RL; also to watch the robot move |
| `sim/rl_residual.py` | **Forward walking only.** Trains / tests the network that corrects the 3D gait on terrain, slopes and pushes, with the command fixed to forward | To make straight walking robust |
| `tests/` | **Automatic checks.** Leg IK vs MuJoCo, every command with the plain gait, terrain, slopes, pushes | After changing code, to make sure nothing broke |

### The files the others use (you don't run these directly)

| File | What it is |
|---|---|
| `sim/bezier3d.py` | **The gait math (paper's 3D Bézier gait).** Turns a command (forward / sideways speed, turn rate) into foot positions in 3D, and a fast exact IK turns those into the 3 joint angles of each leg. |
| `sim/command_env.py` | **The RL training ground** used by both `rl_command.py` and `rl_residual.py`. Runs the 3D gait, lets the network correct it every 1/50 s on random bumps and slopes (with pushes if enabled), and scores how well the robot follows the command. |
| `assets/scene_terrain.xml` | **Bumpy world** for RL (the bumps are re-randomized every episode). |
| `assets/mini_spot.xml` | **The robot itself**: body, legs, motors, sensors, joint limits. |
| `assets/scene.xml` | **The world**: floor, light, sky. It loads the robot. |
| `archive/` | Old, unused code kept for reference. The **2D Bézier gait** and everything built on it: `bezier_helpers.py` (2D curve), `gait_controller.py` (manual walk / turn / circle controller), `gym_class.py` + `rl_ppo.py` (RL stage 1: best 6 gait settings), `residual_env.py` (2D curve reshaping), and their tests. Also `beztest.py`, `mujoco_env.py`, `3dbezierhelpers.py` (its curve now lives in `sim/bezier3d.py`), old `models_walk/` checkpoints. Run them from the repo root as `python -m archive.<name>`. |
| `assets/CAD_Files/` | 3D shapes of the real robot parts (used to draw the robot). |

### Setup files

| File | What it is |
|---|---|
| `docker/Dockerfile` | Recipe for a ready-made Linux box with Python, MuJoCo, PyTorch and RL libraries installed |
| `docker/docker-compose.yml` | Settings for that box (shares this folder and your screen with it) |
| `run.sh` | One command that starts the box and gives you a terminal inside it |

---

## How the files are connected

```
          assets/scene.xml / scene_terrain.xml  ──loads──►  assets/mini_spot.xml
          (the world, flat or bumpy)                        (the robot)
                             ▲
                             │ loads the world + robot
                             │
                     sim/command_env.py   (runs the gait, adds bumps / slopes / pushes, gives the score)
                      ▲              ▲                   │
                      │              │                   │ uses
      sim/rl_command.py              sim/rl_residual.py  ▼
      (every move:                   (forward only)    sim/bezier3d.py
       forward, back,                                  (3D foot curve + leg IK:
       strafe, turn)                                    where each foot goes, when)
```

**Two ways to train, one environment:**

1. **`rl_command.py`** – one network for every command (forward, backward, strafe, turn, stand).
   `drive --plain` also lets you drive the plain gait by hand, without any network.
2. **`rl_residual.py`** – the same environment with the command fixed to forward at 0.15 m/s:
   a network that only learns to keep straight walking robust.

---

## How to run it

### On the Linux PC (Docker – recommended)

First time, or after the `Dockerfile` changes:
```bash
git clone -b rl-gait https://github.com/thejaswidev/mini-SPOT
cd mini-SPOT
docker compose -f docker/docker-compose.yml build
```

Every time:
```bash
./run.sh                    # you are now inside the box
```

Then, inside the box:
```bash
python -m sim.rl_command drive --plain                   # drive the robot (no network needed)
python -m sim.rl_command train                           # train RL
python -m sim.rl_command drive                           # drive the trained network
python -m pytest tests                                   # run the checks
```

### On a Mac (without Docker)

One-time setup:
```bash
python3 -m venv .venv
.venv/bin/pip install mujoco==3.1.6 "numpy<2" gymnasium stable-baselines3 torch pytest
```

Then use `.venv/bin/mjpython` for anything that opens a 3D window (Mac needs it),
and `.venv/bin/python` for everything else:
```bash
.venv/bin/mjpython -m sim.rl_command drive --plain
.venv/bin/python -m sim.rl_command train
.venv/bin/python -m pytest tests
```

> Run commands from the `mini-SPOT` folder, in a normal terminal (you need to type into it).

---

## How the walking works

**One foot, one cycle** (repeats over and over):

```
  SWING (foot in the air, half the time)        STANCE (foot on the ground, half the time)
  foot lifts and moves along a Bézier curve      foot pushes back along a shallow sine dip
                                                 → this push moves the body
```

The swing follows a smooth **12-point Bézier curve** (from the D²-GMBC paper), so the leg moves
smoothly instead of jerking. Two numbers shape it: **ψ** = how high the foot lifts, **δ** = how far
the foot presses into the ground during stance.

**Four legs together** – the gait is a **trot**: diagonal pairs (front-right + rear-left, then
front-left + rear-right) move together.

**Which way to go** – a command (forward / sideways speed, turn rate) decides the direction of each
foot's loop:

- **forward / backward / sideways:** all feet push in the same direction, opposite to where the body should go.
- **turning:** each foot moves along the circle around the body centre, so the body spins on the spot.
- both can be mixed (walk forward while turning).

The foot positions are then turned into the 3 motor angles of each leg by an exact leg **IK**
(`sim/bezier3d.py`), so all 12 motors are used.

---

## How the RL works

### One network for all moves (`rl_command.py`) — the paper's architecture

This follows the D²-GMBC paper (Fig. 2) exactly. The Bézier curve now describes **where the foot
goes in 3D**, and a **command** says which way to go:

```
  command (forward/back speed, sideways speed, turn rate) ───────────────┐
                                                                         ▼
  IMU + gait clock + command ──► network ──► ψ, δ  (curve shape) ──►  3D Bézier gait (sim/bezier3d.py)
                                        └──► Δf    (small foot offsets) ──(+)──► IK ──► 12 servos
```

- **One network learns every move:** forward, backward, strafe left / right, turn left / right
  and stand. During training the command changes every 3–6 s, so it also learns switching.
- **Network outputs (14), all small, 0 = the plain paper gait:** ψ foot clearance (0.04 ± 0.02 m),
  δ stance depth (0.005 ± 0.005 m), and a foot offset per leg in x, y, z (± 1.5 cm).
- **Inputs:** IMU (tilt + rotation speed), gait clock, the command, last action — like the paper.
- **Trained on** random bumps 0–6 cm, uphill/downhill ±15°, side slopes ±8° (`assets/scene_terrain.xml`;
  slopes are made by tilting gravity, the same physics as an incline), and different friction, body mass,
  motor strength, sensor noise and servo delay. Pushes are off by default.
- **Score:** ➕ moving at the commanded speed / turn rate, body level, not falling; ➖ big or jerky changes.

The plain paper gait (no network) already follows every command on flat ground — the network's
job is to keep that working on bumps and slopes:

| Command | Asked | Plain paper gait got |
|---|---|---|
| forward / backward | ±0.15 m/s | +0.14 / −0.18 m/s |
| strafe left / right | ±0.10 m/s | +0.10 / −0.10 m/s |
| turn left / right | ±0.5 rad/s | +0.57 / −0.58 rad/s |

```bash
python -m sim.rl_command drive --plain            # drive the plain paper gait in 3D (no training needed)
python -m sim.rl_command train                    # train the network (5 M steps, 8 parallel sims)
python -m sim.rl_command drive                    # drive the trained network: w a s d = move, q e = turn, x = stop
python -m sim.rl_command eval                     # plain gait vs gait + network for every command (flat)
python -m sim.rl_command eval --scenario mixed    # same on bumps + slopes
```

On a Mac use `.venv/bin/mjpython` for `drive` and `.venv/bin/python` for the rest.

### Forward walking only (`rl_residual.py`)

The same environment and network as above, but the command is always **forward at 0.15 m/s**.
The network only has to learn one thing: keep straight walking going on bumps, slopes and pushes.

```bash
python -m sim.rl_residual train                     # 3 M steps, 8 parallel sims, saves models/residual3d_ppo.zip
python -m sim.rl_residual eval                      # plain gait vs gait + network on flat / bumps / slopes / pushes
python -m sim.rl_residual eval --scenarios uphill15 bumps6 --episodes 20
python -m sim.rl_residual watch --scenario uphill15 # watch it in 3D (on a Mac: .venv/bin/mjpython -m sim.rl_residual watch ...)
python -m sim.rl_residual watch --scenario bumps6 --plain   # same, plain gait (no network)
python -m sim.rl_residual eval --model models/residual3d_ckpt_2000000_steps   # test a checkpoint (saved every 250 k steps)
```

Scenarios: `flat`, `bumps4`, `bumps6`, `uphill10`, `uphill15`, `downhill15`, `side8`, `push80`,
`mixed` (everything at once, like training). Stopping training with Ctrl+C saves to
`models/residual3d_ppo_interrupted` — the finished model is not overwritten.

Models trained with the old 2D version (`models/residual_ppo.zip`) don't fit the new network
inputs / outputs; the old code is in `archive/residual_env.py`.

---

## The robot hardware

| Part | Details |
|---|---|
| Brain (microcontroller) | ESP32 |
| Motor driver | PCA9685 (controls up to 16 servos) |
| Motors | 12 servos, 3 per leg |
| Battery | LiPo |
| Planned onboard computer | NVIDIA Jetson Nano |

Each leg has 3 motors: **sideways hip**, **shoulder**, **knee** — all 12 are in the simulation,
and the 3D Bézier gait + IK drives all 12.

Sensors: BNO055 IMU (tilt, rotation speed) — simulated now,
on the robot later.

---

## Roadmap

| Step | What | Status |
|---|---|---|
| 1 | Docker setup + GitHub | ✅ Done |
| 2 | Robot model in the simulation | ✅ Done |
| 3 | Smooth leg curves (2D Bézier) + walk / trot, turn, circles (now in `archive/`) | ✅ Done |
| 4 | 3D Bézier gait + leg IK: forward, backward, strafe, turn on command | ✅ Done |
| 5 | RL: one network for all commands on bumps, slopes and pushes (sim) | 🔄 In progress |
| 6 | RL: robust forward walking (sim) | 🔄 In progress |
| 7 | Real slopes / stairs in the simulation (not only tilted gravity) | ⬜ Next |
| 8 | Run it on the real robot (Jetson runs the network, ESP32 the servos) | ⬜ Later |

---

*Bachelor's project — Robotics Engineering, Germany*
