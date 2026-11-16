"""Run the real-robot deploy code (sim/rl_deploy.py) against MuJoCo instead of the ESP32.

    rl_deploy.run() + GaitPolicy  --servo degrees (SERVO_MAP)-->  MujocoLink  --> joint angles --> MuJoCo
                                  <--BNO055-style gravity + gyro--             <-- trunk orientation / qvel

Unlike `python -m sim.rl_command drive`, this goes through the exact deploy path: the stand phase, rate
limit, tilt / IMU-timeout safety stops, the SERVO_MAP conversion (channel, sign, zero, min/max clipping)
and the IMU conversion (GRAVITY_SIGN, GYRO_IN_DEG, IMU_TO_BODY). A wrong sign or zero in SERVO_MAP, or a
wrong IMU mounting, shows up here as a robot that stands or walks badly.

    python -m sim.rl_deploy_mujoco                         # trained network, flat ground, 3D viewer
    python -m sim.rl_deploy_mujoco --plain                 # plain paper gait, no network
    python -m sim.rl_deploy_mujoco --scenario bumps6       # any scenario from sim/rl_command.py
    python -m sim.rl_deploy_mujoco --no-viewer             # headless
    python -m sim.rl_deploy_mujoco --imu-delay 2           # IMU readings arrive 2 control steps late
    python -m sim.rl_deploy_mujoco --gamepad               # Xbox controller (sim/gamepad.py)

Drive commands (type + Enter): w forward  s backward  a strafe left  d strafe right
                               q turn left  e turn right  x stop       Ctrl+C = stand, then quit
"""
from collections import deque

import numpy as np
import mujoco

import sim.rl_deploy as deploy
from sim.command_env import CommandGaitEnv
from sim.rl_command import SCENARIOS


class MujocoLink:
    """Same interface as SerialLink: read_imu(), send_servos(deg), read_command(), relax(), close()."""

    def __init__(self, policy, scenario="flat", viewer=True, randomize=False, imu_delay=0, seed=None):
        self.env = CommandGaitEnv(randomize=randomize, max_episode_s=1e9, **SCENARIOS[scenario])
        self.env.reset(seed=seed)                                 # builds terrain / slope, stands, settles
        self.model, self.data = self.env.model, self.env.data
        self.names = policy.names                                 # actuator order of the deploy code
        self.n_sim = max(1, round(policy.control_dt / self.model.opt.timestep))
        self.imu_buf = deque(maxlen=imu_delay + 1)
        self.viewer = None
        if viewer:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    # ------------------------------------------------ IMU: simulate the BNO055 in its own axes
    def _bno055(self):
        R = self.data.xmat[self.env.trunk].reshape(3, 3)
        g = self.model.opt.gravity
        down_body = R.T @ (g / np.linalg.norm(g))
        w_body = self.data.qvel[self.env.root_dof + 3:self.env.root_dof + 6]
        body_to_imu = deploy.IMU_TO_BODY.T                        # inverse of the mounting rotation
        grav_raw = body_to_imu @ (9.81 * down_body / deploy.GRAVITY_SIGN)   # BNO055: +9.81 "up" at rest
        gyro_raw = body_to_imu @ (np.degrees(w_body) if deploy.GYRO_IN_DEG else w_body)
        return grav_raw, gyro_raw

    def read_imu(self):
        self.imu_buf.append(self._bno055())
        return deploy.imu_to_body(*self.imu_buf[0])               # oldest = delayed reading

    # ------------------------------------------------ servos: undo SERVO_MAP, step the sim one control period
    def _deg_to_joints(self, deg):
        q = np.zeros(len(self.names))
        for i, name in enumerate(self.names):
            s = deploy.SERVO_MAP[name]
            q[i] = np.radians((deg[s["channel"]] - s["zero_deg"]) / s["sign"])
        return q

    def send_servos(self, deg):
        self.data.ctrl[:] = self._deg_to_joints(deg)
        for _ in range(self.n_sim):
            mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise KeyboardInterrupt                           # viewer closed -> stop like Ctrl+C
            self.viewer.sync()

    def read_command(self):
        return None                                               # keyboard, handled by deploy.run

    def relax(self):
        """PWM off: position servos lose their stiffness; let the robot settle for 1 s."""
        self.model.actuator_gainprm[:, 0] = 0.0
        self.model.actuator_biasprm[:, 1:3] = 0.0
        print("R (relax)")
        for _ in range(int(deploy.CONTROL_HZ)):
            self.send_servos(np.zeros(len(deploy.SERVO_MAP)))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        self.env.close()


def main():
    parser = deploy.parse_args(__doc__)
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="flat", help="terrain / slope / pushes")
    parser.add_argument("--randomize", action="store_true", help="random friction, mass, motor strength")
    parser.add_argument("--imu-delay", type=int, default=0, help="IMU latency in control steps")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()                                    # --dry-run is accepted but not needed here
    policy = deploy.GaitPolicy(args.model, args.plain)
    link = MujocoLink(policy, args.scenario, not args.no_viewer, args.randomize, args.imu_delay, args.seed)
    deploy.run(link, policy, deploy.make_gamepad(args))


if __name__ == "__main__":
    main()
