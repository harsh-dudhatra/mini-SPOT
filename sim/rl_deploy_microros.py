"""Same as sim/rl_deploy.py (trained policy or plain 3D Bezier gait on the real robot), but the ESP32
talks micro-ROS instead of plain serial lines. Runs on the Jetson Nano (ROS 2 + rclpy installed).

    Jetson Nano                                                          ESP32 (micro-ROS firmware)
      this node  --/mini_spot/servo_cmd  (std_msgs/Float32MultiArray)-->  12 servo angles [deg], channel order
                 <--/mini_spot/imu       (sensor_msgs/Imu)-------------   BNO055 orientation + gyro
                 --/mini_spot/servo_enable (std_msgs/Bool)------------->  False = relax (PWM off)
                 <--/cmd_vel              (geometry_msgs/Twist)  optional: (vx, vy, wz) from any ROS node
         ^
         micro_ros_agent (serial transport) bridges ROS 2 <-> ESP32:
           ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -b 115200

    python3 -m sim.rl_deploy_microros                  # trained network (models/command_ppo)
    python3 -m sim.rl_deploy_microros --plain          # plain paper gait, no network
    python3 -m sim.rl_deploy_microros --dry-run        # no ESP32 / agent: fake IMU, prints the frames

Keyboard commands are the same as sim/rl_deploy.py; a /cmd_vel message overrides them.
The servo mapping, IMU axes, safety limits and the control loop all come from sim/rl_deploy.py:
fill in its TODO(hardware) settings first, plus the TODO(hardware) topics / IMU source below.
"""
import numpy as np

from sim.rl_deploy import (IMU_TO_BODY, DryRunLink, GaitPolicy, check_hardware, imu_to_body,
                           make_gamepad, parse_args, run)

# ================================================================
#  TODO(hardware): match the ESP32 micro-ROS firmware
# ================================================================
NODE_NAME = "mini_spot_rl"
SERVO_TOPIC = "/mini_spot/servo_cmd"       # Float32MultiArray, 12 values (allocate 12 floats on the ESP32)
ENABLE_TOPIC = "/mini_spot/servo_enable"   # Bool
IMU_TOPIC = "/mini_spot/imu"               # sensor_msgs/Imu
CMD_TOPIC = "/cmd_vel"                     # geometry_msgs/Twist: linear.x, linear.y, angular.z
IMU_SOURCE = "orientation"                 # "orientation": BNO055 fused quaternion (best, ignores shaking)
                                           # "accel": linear_acceleration (raw accelerometer incl. gravity)


def gravity_from_quat(x, y, z, w):
    """Down direction (0, 0, -1) of the world, seen in the IMU's own axes (quaternion = IMU -> world)."""
    # third row of the rotation matrix, negated: R^T @ (0, 0, -1)
    return -np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])


class MicroRosLink:
    def __init__(self):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import Imu
        from std_msgs.msg import Bool, Float32MultiArray

        self.rclpy, self.Float32MultiArray, self.Bool = rclpy, Float32MultiArray, Bool
        rclpy.init()
        self.node = rclpy.create_node(NODE_NAME)
        # best effort + depth 1: what micro-ROS handles best, and only the newest value matters
        self.servo_pub = self.node.create_publisher(Float32MultiArray, SERVO_TOPIC, qos_profile_sensor_data)
        self.enable_pub = self.node.create_publisher(Bool, ENABLE_TOPIC, 10)
        self.node.create_subscription(Imu, IMU_TOPIC, self._on_imu, qos_profile_sensor_data)
        self.node.create_subscription(Twist, CMD_TOPIC, self._on_cmd, 10)
        self.imu = None
        self.cmd = None
        self.enable_pub.publish(Bool(data=True))

    def _on_imu(self, msg):
        if IMU_SOURCE == "orientation":
            o = msg.orientation
            grav = IMU_TO_BODY @ gravity_from_quat(o.x, o.y, o.z, o.w)
            w = msg.angular_velocity                    # sensor_msgs/Imu is rad/s by definition
            gyro = IMU_TO_BODY @ np.array([w.x, w.y, w.z])
            self.imu = (grav / max(np.linalg.norm(grav), 1e-6), gyro)
        else:
            a, w = msg.linear_acceleration, msg.angular_velocity
            # accelerometer at rest reads +9.81 "up", same as the BNO055 gravity vector
            grav, _ = imu_to_body([a.x, a.y, a.z], [0.0, 0.0, 0.0])
            self.imu = (grav, IMU_TO_BODY @ np.array([w.x, w.y, w.z]))

    def _on_cmd(self, msg):
        self.cmd = (msg.linear.x, msg.linear.y, msg.angular.z)

    def read_imu(self):
        """Process waiting ROS messages, return the IMU reading received since the last call."""
        for _ in range(10):                             # spin_once handles one message per call
            self.rclpy.spin_once(self.node, timeout_sec=0.0)
        imu, self.imu = self.imu, None
        return imu

    def send_servos(self, deg):
        self.servo_pub.publish(self.Float32MultiArray(data=[float(d) for d in deg]))

    def read_command(self):
        cmd, self.cmd = self.cmd, None
        return cmd

    def relax(self):
        self.enable_pub.publish(self.Bool(data=False))

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


def main():
    args = parse_args(__doc__).parse_args()
    check_hardware(args.dry_run)
    policy = GaitPolicy(args.model, args.plain)
    link = DryRunLink() if args.dry_run else MicroRosLink()
    run(link, policy, make_gamepad(args))


if __name__ == "__main__":
    main()
