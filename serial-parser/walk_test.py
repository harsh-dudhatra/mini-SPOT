# simple trot for ~10 s: diagonal legs (FR+RL, FL+RR) move together
# each leg: thigh swings back and forth around home (sine),
#           calf lifts the foot only while the thigh swings forward
# same serial protocol and 50 Hz loop as serialParser.py
# TEST IN THE AIR FIRST (Spot on a box, feet off the ground)

import math
import sys
import time

import serial

PORT = "/dev/ttyUSB0"
BAUD = 115200

RATE_HZ = 50
CYCLE_TIME = 1.0 / RATE_HZ

WALK_TIME = 20.0  # seconds of walking
RAMP_TIME = 1.0    # grow the steps in at the start, shrink them at the end
STEP_PERIOD = 1.0  # seconds for one full step of one leg

# small on purpose, all inside MIN_US/MAX_US in main.cpp (RR is only home +-100)
THIGH_AMP = 80  # us, thigh swings home +- this
# us, calf bends this much to lift the foot, per leg: FR, FL, RR, RL
# RR only 100: its limits are still home +-100 until the leg is repaired
CALF_AMP = [150, 150, 100, 150]

HOME_US = [1525, 1150, 1500,   # FR
           1650, 1700, 1500,   # FL
           1600, 1225, 1500,   # RR
           1500, 1700, 1500]   # RL

# legs in the same order as HOME_US: FR, FL, RR, RL
PHASE = [0.0, 0.5, 0.5, 0.0]  # FR+RL together, FL+RR half a step later
# LAB: +1 or -1 so that "+" means thigh forward / foot up on every leg.
# right and left servos are mirrored, so these are a guess: check in the air, flip if wrong
THIGH_SIGN = [-1, +1, -1, +1]
CALF_SIGN = [-1, +1, -1, +1]


def pose_at(t):
    """12 pulses (us) for time t seconds after the start of the walk."""
    # 0 -> 1 during the first RAMP_TIME, 1 -> 0 during the last RAMP_TIME
    ramp = min(1.0, t / RAMP_TIME, (WALK_TIME - t) / RAMP_TIME)
    ramp = max(0.0, ramp)

    pulses = list(HOME_US)  # shoulders stay at home
    for leg in range(4):
        angle = 2 * math.pi * (t / STEP_PERIOD + PHASE[leg])
        thigh = -math.cos(angle)       # -1 -> +1 in the first half (swing), back in the second (stance)
        lift = max(0.0, math.sin(angle))  # > 0 only in the first half, foot on the ground otherwise

        pulses[3 * leg + 1] += round(THIGH_SIGN[leg] * ramp * THIGH_AMP * thigh)
        pulses[3 * leg + 2] += round(CALF_SIGN[leg] * ramp * CALF_AMP[leg] * lift)
    return pulses


def send(ser, pulses):
    ser.reset_input_buffer()
    ser.write((",".join(str(us) for us in pulses) + "\n").encode())
    reply = ser.readline().decode(errors="replace").strip()
    if reply.startswith("Err"):
        print("ESP32 rejected it:", reply)


try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
except serial.SerialException as e:
    print(f"Could not open {PORT}: {e}")
    sys.exit(1)

try:
    time.sleep(2)  # opening the port resets the ESP32, wait for setup()
    ser.reset_input_buffer()
    ser.timeout = 0.1

    print(f"walking for {WALK_TIME} s, Ctrl+C to stop")
    start = time.perf_counter()
    next_tick = start
    while True:
        t = time.perf_counter() - start
        if t > WALK_TIME:
            break
        send(ser, pose_at(t))

        next_tick += CYCLE_TIME
        wait = next_tick - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        else:
            next_tick = time.perf_counter()  # fell behind, don't burst

except KeyboardInterrupt:
    print("stopped")

finally:
    # stand still at home; a few times, the firmware only moves MAX_STEP per command
    for _ in range(5):
        send(ser, HOME_US)
        time.sleep(CYCLE_TIME)
    print("back at home")
    ser.close()
