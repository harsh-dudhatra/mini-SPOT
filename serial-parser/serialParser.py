# bridge between the RL code and the ESP32
# send:    12 pulses (us), comma separated, '\n' at the end
# receive: "OK p0 ... p11", "Err: ..." and "I,gx,gy,gz,wx,wy,wz" (IMU, 50 Hz)

import sys
import time
import math
import serial

PORT = "/dev/ttyUSB0"
BAUD = 115200  # same as Serial.begin() in main.cpp

RATE_HZ = 50
CYCLE_TIME = 1.0 / RATE_HZ  # 20 ms

# sim pose that matches HOME_US on the real robot, from the MuJoCo
# shoulder, thigh, calf per leg
SIM_HOME = [0.0, 0.687, -1.0,   # FR
            0.0, 0.687, -1.0,   # FL
            0.0, 0.687, -1.0,   # RR
            0.0, 0.687, -1.0]   # RL

HOME_US = [1525, 1150, 1500,   # FR
           1650, 1700, 1500,   # FL
           1600, 1225, 1500,   # RR
           1500, 1700, 1500]   # RL

MIN_US = [1195,  740,  970,
          1050, 1100, 1150,
          1130, 1125, 1400,
           520, 1200, 1110]
MAX_US = [1975, 1840, 1870,
          2550, 1950, 2130,
          2510, 1325, 1600,
          1970, 2290, 1990]

SIGN = [+1, -1, -1,   # FR
        +1, +1, +1,   # FL
        +1, -1, -1,   # RR
        +1, +1, +1]   # RL

# 180 deg servo, 500..2500 us -> 2000 us per pi rad.
US_PER_RAD = 2000 / math.pi

def sim_to_angles(sim_angles):
    """12 sim angles in rad (shoulder, thigh, calf per leg: FR, FL, RR, RL) -> 12 angles, 0 = home"""
    if len(sim_angles) != 12:
        raise ValueError(f"need 12 sim angles, got {len(sim_angles)}")
    return [sim_angles[i] - SIM_HOME[i] for i in range(12)]

def rad_to_pulse(angles):
    """12 joint angles in rad (0 = home) -> 12 pulses in us, clamped to the servo limits"""
    pulses = []
    for i in range(12):
        us = HOME_US[i] + SIGN[i] * angles[i] * US_PER_RAD
        us = max(MIN_US[i], min(MAX_US[i], round(us)))
        pulses.append(us)
    return pulses


def read_line(ser):
    """one line from the ESP32, '' on timeout"""
    raw = ser.readline()
    # boot output can have non-text bytes
    return raw.decode(errors="replace").strip()

latest_imu = None  # newest [gx, gy, gz, wx, wy, wz], None until the first IMU line


def parse_imu(line):
    """'I,gx,gy,gz,wx,wy,wz' -> [gx, gy, gz, wx, wy, wz], None if not a good IMU line"""
    if not line.startswith("I,"):
        return None
    parts = line[2:].split(",")
    if len(parts) != 6:
        return None  # half a line, e.g. right after boot
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    # gyro deg/s -> rad/s
    values[3:] = [math.radians(w) for w in values[3:]]
    return values


def read_imu(ser):
    """read all waiting lines, return the newest IMU values (None if none came yet)"""
    global latest_imu
    while ser.in_waiting:
        imu = parse_imu(read_line(ser))
        if imu is not None:
            latest_imu = imu
    return latest_imu

def open_port():
    """open the port, wait for the ESP32 boot, return ser"""
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
    except serial.SerialException as e:
        print(f"Could not open {PORT}: {e}")
        print("Plugged in? Right port (ls /dev/ttyUSB*)? PlatformIO monitor closed?")
        sys.exit(1)

    # opening the port resets the ESP32, wait for setup()
    time.sleep(2)

    # print boot output until it goes quiet
    while True:
        line = read_line(ser)
        if line == "":
            break
        print("boot:", line)

    # replies come within a few ms now
    ser.timeout = 0.1
    return ser


def send_angles(ser, sim_angles):
    """send 12 sim angles (rad) once, return the ESP32 reply ('' on timeout)"""
    global latest_imu
    command = ",".join(str(us) for us in rad_to_pulse(sim_to_angles(sim_angles))) + "\n"
    # read old lines first (not reset_input_buffer, that would throw away IMU lines)
    read_imu(ser)
    ser.write(command.encode())

    # IMU lines can come before the reply, keep reading until OK or Err
    while True:
        reply = read_line(ser)
        if reply == "":
            print("No reply within 0.1 s")
            return reply
        imu = parse_imu(reply)
        if imu is not None:
            latest_imu = imu
        elif reply.startswith("Err"):
            print("ESP32 rejected it:", reply)
            return reply
        elif reply.startswith("OK"):
            return reply


if __name__ == "__main__":
    ser = open_port()
    print(f"holding home pose at {RATE_HZ} Hz, Ctrl+C to stop")
    try:
        loop_count = 0
        # start the clock here, after the boot wait, or the loop sends a burst to catch up
        time_counter = time.perf_counter()

        while True:
            reply = send_angles(ser, SIM_HOME)
            if loop_count % RATE_HZ == 0:
                # print once a second, not every reply
                print("reply:", reply)
            loop_count += 1

            now = time.perf_counter()
            elapsed = now - time_counter
            if elapsed < CYCLE_TIME:
                time.sleep(CYCLE_TIME - elapsed)
                time_counter += CYCLE_TIME
            else:
                # fell behind: restart the clock, no catch-up burst
                time_counter = now

    except KeyboardInterrupt:
        print("stopped")

    finally:
        # runs on Ctrl+C and on crashes
        ser.close()
