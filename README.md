# Mini-SPOT

A miniature quadruped robot inspired by Boston Dynamics Spot.

The goal is a fully autonomous legged robot: simulation, kinematics, gait control, and real hardware deployment.

---

## System overview

The project has two parts, split by where the code runs:

```
┌───────────────────────────────┐   USB serial   ┌──────────────────────────────┐
│ Part 1: Jetson Nano / laptop  │ ─────────────▶ │ Part 2: ESP32                │
│ simulation, gait, RL, ROS2    │ ◀───────────── │ servos, sensors, micro-ROS   │
└───────────────────────────────┘                └──────────────────────────────┘
  docker/  assets/  sim/  tests/                   firmware/
```

| Part | Runs on | Folders | Docs |
|---|---|---|---|
| **1. Simulation & control** | Jetson Nano / laptop (Docker) | `docker/`, `assets/`, `sim/`, `tests/` | this README |
| **2. Firmware** | ESP32 DevKit | `firmware/` | [firmware/README.md](firmware/README.md) |

Wiring diagram: [docs/wiring_devkit.svg](docs/wiring_devkit.svg)

---

## Hardware

| Component | Details |
|---|---|
| Microcontroller | ESP32 DevKit (ESP32-WROOM-32U, CP2102 USB), replacing the earlier ESP32-S2 |
| Servo controller | PCA9685 (I2C, 16-channel PWM) |
| Servos | 12 total, 3 per leg (HAA, HFE, KFE joints) |
| IMU | BNO055 (I2C), being wired |
| Power | 7.4V LiPo → HW-439 buck (servo power) → LM1085 3.3V (logic) |
| Onboard compute | NVIDIA Jetson Nano |

4 legs × 3 joints = 12 servos total. Each leg has full 3-DOF: Hip Abduction/Adduction (HAA, "shoulder"), Hip Flexion/Extension (HFE, "thigh"), Knee Flexion/Extension (KFE, "calf").

HAA joints are currently locked in simulation, leaving 8 active DOF (HFE + KFE per leg). This simplifies IK to a clean 2D problem per leg.

---

## Roadmap

### Part 1: simulation & control

| Step | Focus | Status |
|---|---|---|
| 1 | Docker environment + GitHub setup | ✅ Complete |
| 2 | MJCF robot model in MuJoCo | ✅ Complete |
| 3 | Inverse kinematics, Bezier trajectories, gait scheduler | 🔄 In progress |
| 4 | Full locomotion controller (Mini-SPOT walks in sim) | 🔄 In progress |
| 5 | Serial bridge: joint angles → pulses → ESP32 | ⬜ Pending |
| 6 | ROS2 on the Jetson Nano, gamepad control | ⬜ Pending |

### Part 2: firmware (details in [firmware/README.md](firmware/README.md))

| Step | Focus | Status |
|---|---|---|
| 1 | PlatformIO project, pin map per board | ✅ Complete |
| 2 | Servo command receiver (12 pulses per line, limits, step limit) | ✅ Written, robot test pending |
| 3 | Limit finder, home + limit measurement | ✅ Done except RR leg |
| 4 | Move to ESP32 DevKit, new wiring | 🔄 In progress |
| 5 | Watchdog, BNO055 IMU | ⬜ Pending |
| 6 | micro-ROS | ⬜ Pending |

---

## Part 1: simulation

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- Git

### Setup

**Linux:**
```bash
git clone https://github.com/harsh-dudhatra/mini-SPOT
cd mini-SPOT
chmod +x run.sh
./run.sh
```

**macOS:**
Install [XQuartz](https://www.xquartz.org) first, then:
```bash
xhost +localhost
git clone https://github.com/harsh-dudhatra/mini-SPOT
cd mini-SPOT
chmod +x run.sh
./run.sh
```

**Windows:**
Install [VcXsrv](https://sourceforge.net/projects/vcxsrv/), launch XLaunch with "Disable access control" checked, then:
```bash
git clone https://github.com/harsh-dudhatra/mini-SPOT
cd mini-SPOT
chmod +x run.sh
./run.sh
```

### Run the simulation

```bash
# Inside the container
python sim/mujoco_env.py
```

MuJoCo viewer opens with Mini-SPOT standing on a checkered ground plane.
Use left mouse to orbit, right mouse to pan, scroll to zoom.

### Robot model

The MJCF model (`assets/mini_spot.xml`) describes the full robot:

- Rectangular torso with freejoint (6 DOF in world)
- 4 legs × 3 bodies (hip, thigh, calf) with real STL mesh geometry
- 8 active position actuators (HFE + KFE per leg, HAA locked)
- IMU sensor site on trunk (accelerometer + gyro)
- Foot force sensor sites on each calf (contact detection)
- Joint limits: HFE [-0.785, 3.14] rad, KFE [-2.44, -0.916] rad
- Keyframes: `start` (crouched stand) and `home` (extended)

The scene file (`assets/scene.xml`) includes the robot and adds the world:
- Checkered ground plane with proper friction
- Skybox gradient
- Directional lighting with shadows

### Control architecture

```
VelocityCommand (vx, vy, yaw)
    └── GaitScheduler        → phase and swing/stance per leg
          └── BezierTrajectory → foot target position (x, z)
                └── IKSolver   → joint angles (θ_HFE, θ_KFE)
                      └── MuJoCo actuators / ESP32 servos
```

Physics runs at 1000Hz. Controller runs at 100Hz (every 10 physics steps).

### Sensors: simulation vs. hardware

| Sensor | Location | Data | Hardware |
|---|---|---|---|
| Accelerometer | Trunk | Linear acceleration (m/s²) | BNO055 on the ESP32 I2C bus |
| Gyroscope | Trunk | Angular velocity (rad/s) | BNO055 on the ESP32 I2C bus |
| Foot force | Each calf | 3-axis contact force (N) | – (simulation only) |

---

## Repo structure

```
mini-SPOT/
├── README.md                   ← this file (overview + Part 1)
├── docs/
│   └── wiring_devkit.svg       ← wiring diagram (ESP32 DevKit)
│
│   Part 1: simulation & control (Jetson / laptop)
├── run.sh                      ← one-command start (X11 + container)
├── docker/
│   ├── Dockerfile              ← Python 3.11, MuJoCo, all dependencies
│   └── docker-compose.yml      ← container config, volume mounts, display
├── assets/
│   ├── mini_spot.xml           ← MuJoCo MJCF robot model
│   ├── scene.xml               ← simulation world (ground, lighting, skybox)
│   └── CAD_Files/              ← STL meshes + STP models of all printed parts
├── sim/
│   └── mujoco_env.py           ← MuJoCo simulation runner
│
│   Part 2: firmware (ESP32)
└── firmware/                   ← PlatformIO project, see firmware/README.md
```

---

## Tech stack

| Tool | Role |
|---|---|
| Python 3.11 | Simulation and control |
| MuJoCo 3.1.6 | Physics simulation |
| MJCF XML | Robot model format |
| NumPy / SciPy | Kinematics and math |
| Matplotlib | Visualization and debugging |
| Docker | Reproducible environment |
| C++ (Arduino) + PlatformIO | ESP32 firmware |
| Adafruit PWM Servo Driver / BNO055 libraries | PCA9685 and IMU drivers |
| NVIDIA Jetson Nano | Onboard compute |

---

*Bachelor's project, Robotics Engineering, Germany*
