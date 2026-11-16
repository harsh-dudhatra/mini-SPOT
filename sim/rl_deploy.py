"""Run a trained policy (or the plain 3D Bezier gait) on the real Mini-SPOT. Runs on the Jetson Nano.

    Jetson Nano (this script)  <--- serial (USB / UART) --->  ESP32  --> PCA9685 --> 12 servos
                                                                     <-- BNO055 IMU

  Every 20 ms (50 Hz, the same rate as training):
    IMU from the ESP32 --> obs --> policy --> psi, delta, foot offsets
        --> 3D Bezier gait + IK (sim/bezier3d.py) --> 12 joint angles [rad, simulation convention]
        --> servo mapping (SERVO_MAP) --> one serial line --> ESP32 moves the servos

    python -m sim.rl_deploy --dry-run                        # no ESP32: level fake IMU, prints the frames
    python -m sim.rl_deploy --port /dev/ttyUSB0              # trained network (models/command_ppo)
    python -m sim.rl_deploy --port /dev/ttyUSB0 --plain      # plain paper gait, no network
    python -m sim.rl_deploy --model models/command_ckpt_249998_steps

Drive commands (type + Enter, or --gamepad: see sim/gamepad.py): w forward  s backward  a strafe left  d strafe right
                               q turn left  e turn right  x stop       Ctrl+C = stand, then quit

Serial protocol (ASCII lines, readable in a serial monitor; TODO(hardware): match the ESP32 code):
    Jetson -> ESP32   S,<deg ch0>,<deg ch1>,...,<deg ch11>\\n     servo angles in channel order
                      R\\n                                         relax: servos off
    ESP32 -> Jetson   I,<gx>,<gy>,<gz>,<wx>,<wy>,<wz>\\n         BNO055 gravity vector + gyro

Before the first real run: fill in everything marked TODO(hardware), set HARDWARE_CONFIGURED = True,
and test with the robot on a stand (feet off the ground). The same controller is used over micro-ROS
by sim/rl_deploy_microros.py.
"""
import argparse
import select
import sys
import time

import numpy as np
import mujoco

from sim.bezier3d import DELTA_DEFAULT, LEG_NAMES, PSI_DEFAULT, Bezier3DGait
from sim.mujoco_driver import MujocoDriver
from sim.command_env import CMD_SCALE, COMMANDS, DELTA_SCALE, FLAT_XML, PSI_SCALE, RESIDUAL_LIMIT
from sim.rl_command import KEYS, MODEL_PATH, TYPICAL, load_policy
from sim.gamepad import QuitRequested, open_gamepad

# ================================================================
#  TODO(hardware): fill these in for your robot
# ================================================================
HARDWARE_CONFIGURED = False        # set True once SERVO_MAP / IMU / serial below are checked

SERIAL_PORT = "/dev/ttyUSB0"       # ESP32 over USB; Jetson UART pins: /dev/ttyTHS1
BAUD = 115200

# One entry per joint (simulation names). servo_deg = zero_deg + sign * degrees(joint angle in rad)
#   channel   PCA9685 output the servo is plugged into
#   sign      +1 / -1: does the servo turn the same way as the simulated joint?
#   zero_deg  servo angle at which the simulated joint angle is 0
#             (hip 0 = leg straight down sideways; thigh 0 = leg straight down; knee 0 = leg straight)
#   min_deg / max_deg   mechanical limits of that servo on the robot
SERVO_MAP = {
    "FR_hip":   dict(channel=0,  sign=+1, zero_deg=90.0, min_deg=0.0, max_deg=180.0),
    "FR_thigh": dict(channel=1,  sign=+1, zero_deg=90.0, min_deg=0.0, max_deg=180.0),
    "FR_calf":  dict(channel=2,  sign=+1, zero_deg=90.0, min_deg=0.0, max_deg=180.0),
    "FL_hip":   dict(channel=3,  sign=+1, zero_deg=90.0, min_deg=0.0, max_deg=180.0),
    "FL_thigh": dict(channel=4,  sign=+1, zero_deg=90.0, min_deg=0.0, max_deg=180.0),
    "FL_calf":  dict(channel=5,  sign=+1, zero_deg=90.0, min_deg=0.0, max_deg=180.0),
    "RR_hip":   dict(channel=6,  sign=+1, zero_deg=90.0, min_deg=0.0, max_deg=180.0),
    "RR_thigh": dict(channel=7,  sign=+1, zero_deg=90.0, min_deg=0.0, max_deg=180.0),
    "RR_calf":  dict(channel=8,  sign=+1, zero_deg=90.0, min_deg=0.0, max_deg=180.0),
    "RL_hip":   dict(channel=9,  sign=+1, zero_deg=90.0, min_deg=0.0, max_deg=180.0),
    "RL_thigh": dict(channel=10, sign=+1, zero_deg=90.0, min_deg=0.0, max_deg=180.0),
    "RL_calf":  dict(channel=11, sign=+1, zero_deg=90.0, min_deg=0.0, max_deg=180.0),
}

# IMU -> robot body frame (x forward, y left, z up, like the simulation).
# Check (after imu_to_body): robot level -> gravity ~ (0, 0, -1), nose down -> x > 0,
# left side down -> y > 0, turning left -> gyro z > 0.
IMU_TO_BODY = np.eye(3)            # rotation from the BNO055 axes to the body axes (how it is mounted)
GRAVITY_SIGN = -1.0                # BNO055 gravity vector reads +9.81 "up" at rest -> -1 makes it point down
GYRO_IN_DEG = True                 # BNO055 default gyro unit is deg/s; the policy wants rad/s

# ================================================================
#  safety
# ================================================================
CONTROL_HZ = 50.0                  # must match training (control_freq)
STAND_S = 2.0                      # hold the stand pose this long before the gait starts
MAX_JOINT_SPEED = 8.0              # rad/s, per-joint rate limit on the commanded angles
TILT_STOP = -0.5                   # gravity z above this (body tilted > 60 deg) -> relax and quit
IMU_TIMEOUT_S = 0.2                # no IMU data for this long -> stand and quit
CMD_LIMITS = np.array([[min(r[0][0] for r in COMMANDS.values()), max(r[0][1] for r in COMMANDS.values())],
                       [min(r[1][0] for r in COMMANDS.values()), max(r[1][1] for r in COMMANDS.values())],
                       [min(r[2][0] for r in COMMANDS.values()), max(r[2][1] for r in COMMANDS.values())]])


class GaitPolicy:
    """Same observation / action / gait as CommandGaitEnv.step, without the simulation."""

    def __init__(self, model_path=MODEL_PATH, plain=False):
        self.model = mujoco.MjModel.from_xml_path(FLAT_XML)       # only for the leg geometry (IK)
        self.driver = MujocoDriver(self.model, LEG_NAMES)
        self.gen = Bezier3DGait(self.driver.legs)
        self.names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                      for i in range(self.model.nu)]
        self.act = (lambda obs: np.zeros(14)) if plain else load_policy(model_path)
        self.control_dt = 1.0 / CONTROL_HZ
        self.phase = 0.0
        self.prev_action = np.zeros(14)
        self.cmd = np.zeros(3)

    def set_command(self, cmd):
        self.cmd = np.clip(np.asarray(cmd, dtype=float), CMD_LIMITS[:, 0], CMD_LIMITS[:, 1])

    def stand(self):
        return self.driver.joint_targets(self.gen.f_stand)

    def step(self, grav, gyro):
        """IMU (body frame) -> 12 joint angles [rad], indexed like the MuJoCo actuators."""
        obs = np.concatenate([grav, gyro,
                              [np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)],
                              self.cmd / CMD_SCALE, self.prev_action]).astype(np.float32)
        action = np.clip(np.asarray(self.act(obs), dtype=np.float64), -1.0, 1.0)
        self.prev_action = action
        self.phase = (self.phase + self.control_dt / self.gen.t_stride) % 1.0
        psi = PSI_DEFAULT + PSI_SCALE * action[0]
        delta = DELTA_DEFAULT + DELTA_SCALE * action[1]
        residual = action[2:].reshape(4, 3) * RESIDUAL_LIMIT
        feet = self.gen.foot_targets(self.phase, self.cmd, psi, delta, residual)
        return self.driver.joint_targets(feet)

    def to_servo_deg(self, q):
        """Joint angles [rad, actuator order] -> servo angles [deg, PCA9685 channel order]."""
        deg = np.zeros(len(SERVO_MAP))
        for name, angle in zip(self.names, q):
            s = SERVO_MAP[name]
            deg[s["channel"]] = np.clip(s["zero_deg"] + s["sign"] * np.degrees(angle),
                                        s["min_deg"], s["max_deg"])
        return deg


def imu_to_body(grav_raw, gyro_raw):
    """BNO055 gravity vector + gyro (sensor axes) -> unit gravity direction + rad/s (body axes)."""
    g = IMU_TO_BODY @ (GRAVITY_SIGN * np.asarray(grav_raw, dtype=float))
    w = np.asarray(gyro_raw, dtype=float)
    if GYRO_IN_DEG:
        w = np.radians(w)
    return g / max(np.linalg.norm(g), 1e-6), IMU_TO_BODY @ w


# ================================================================
#  links to the ESP32: read_imu() -> (grav, gyro) or None, send_servos(deg), read_command(), relax()
# ================================================================
class SerialLink:
    def __init__(self, port=SERIAL_PORT, baud=BAUD):
        import serial                                     # pyserial
        self.ser = serial.Serial(port, baud, timeout=0)
        self.buf = b""
        self.imu = None

    def read_imu(self):
        """Latest IMU line received since the last call (None if nothing new)."""
        self.buf += self.ser.read(self.ser.in_waiting or 1)
        *lines, self.buf = self.buf.split(b"\n")
        new = None
        for line in lines:
            parts = line.decode(errors="ignore").strip().split(",")
            if parts[0] == "I" and len(parts) == 7:       # TODO(hardware): match the ESP32 IMU message
                try:
                    vals = [float(v) for v in parts[1:]]
                except ValueError:
                    continue
                new = imu_to_body(vals[:3], vals[3:])
        return new

    def send_servos(self, deg):
        self.ser.write(("S," + ",".join(f"{d:.1f}" for d in deg) + "\n").encode())

    def read_command(self):
        return None                                       # commands come from the keyboard only

    def relax(self):
        self.ser.write(b"R\n")                           # TODO(hardware): ESP32 turns the PWM off

    def close(self):
        self.ser.close()


class DryRunLink:
    """No hardware: a level, still IMU; prints one servo frame per second."""

    def __init__(self):
        self.n = 0

    def read_imu(self):
        return np.array([0.0, 0.0, -1.0]), np.zeros(3)

    def send_servos(self, deg):
        if self.n % int(CONTROL_HZ) == 0:
            print("S," + ",".join(f"{d:.1f}" for d in deg))
        self.n += 1

    def read_command(self):
        return None

    def relax(self):
        print("R (relax)")

    def close(self):
        pass


# ================================================================
#  control loop (shared with sim/rl_deploy_microros.py)
# ================================================================
def read_keyboard(policy):
    if not select.select([sys.stdin], [], [], 0)[0]:
        return
    line = sys.stdin.readline().strip().lower().replace("_", " ")
    name = KEYS.get(line, line.replace(" ", "_"))
    if name in TYPICAL:
        policy.set_command(TYPICAL[name])
        print(f">> {name}  (vx, vy, wz) = {TYPICAL[name]}")
    elif line:
        print(">> unknown command; use w a s d q e x")


def run(link, policy, gamepad=None):
    dt = policy.control_dt
    max_step = MAX_JOINT_SPEED * dt
    q = policy.stand()
    grav, gyro = np.array([0.0, 0.0, -1.0]), np.zeros(3)
    last_imu = time.time()
    t_start = time.time()
    fallen = False
    print(f">> standing for {STAND_S:.0f} s, then the gait starts (command: stand)")
    print(__doc__.split("Drive commands")[1].split("Serial protocol")[0].strip())
    try:
        while True:
            t0 = time.time()
            imu = link.read_imu()
            if imu is not None:
                grav, gyro = imu
                last_imu = t0
            if t0 - last_imu > IMU_TIMEOUT_S:
                print(">> no IMU data - stopping")
                break
            if grav[2] > TILT_STOP:
                print(">> robot tilted too far - relaxing the servos")
                fallen = True
                break
            cmd = link.read_command()
            if cmd is not None:
                policy.set_command(cmd)
            read_keyboard(policy)
            if gamepad is not None:
                cmd = gamepad.read_command()
                if cmd is not None:
                    policy.set_command(cmd)
                    print(f">> gamepad (vx, vy, wz) = ({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f})")

            target = policy.stand() if t0 - t_start < STAND_S else policy.step(grav, gyro)
            q = q + np.clip(target - q, -max_step, max_step)      # rate limit
            link.send_servos(policy.to_servo_deg(q))
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\n>> Ctrl+C")
    except QuitRequested as e:
        print(f">> {e} - stopping")
    finally:
        if fallen:                                                # don't push against the ground
            link.relax()
        else:
            stand = policy.to_servo_deg(policy.stand())
            for _ in range(int(0.5 * CONTROL_HZ)):                # hold the stand pose for 0.5 s
                link.send_servos(stand)
                time.sleep(dt)
            print(">> stopped in the stand pose")
        link.close()
        if gamepad is not None:
            gamepad.close()


def check_hardware(dry_run):
    if not dry_run and not HARDWARE_CONFIGURED:
        sys.exit("Fill in the TODO(hardware) settings in sim/rl_deploy.py (SERVO_MAP, IMU, serial) and set "
                 "HARDWARE_CONFIGURED = True first. Use --dry-run to test without the robot.")


def parse_args(description):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="trained model or checkpoint")
    parser.add_argument("--plain", action="store_true", help="plain paper gait, no network")
    parser.add_argument("--dry-run", action="store_true", help="no hardware: fake IMU, print servo frames")
    parser.add_argument("--gamepad", nargs="?", const="/dev/input/js0", default=None, metavar="DEVICE",
                        help="drive with an Xbox controller (default device /dev/input/js0)")
    parser.add_argument("--gamepad-mix", action="store_true",
                        help="gamepad: allow combined moves (not seen in training)")
    return parser


def make_gamepad(args):
    return open_gamepad(args.gamepad, args.gamepad_mix) if args.gamepad else None


def main():
    parser = parse_args(__doc__)
    parser.add_argument("--port", type=str, default=SERIAL_PORT, help="serial port of the ESP32")
    parser.add_argument("--baud", type=int, default=BAUD)
    args = parser.parse_args()
    check_hardware(args.dry_run)
    policy = GaitPolicy(args.model, args.plain)
    link = DryRunLink() if args.dry_run else SerialLink(args.port, args.baud)
    run(link, policy, make_gamepad(args))


if __name__ == "__main__":
    main()
