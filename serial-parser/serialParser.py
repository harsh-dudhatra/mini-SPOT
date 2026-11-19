# sends the home pose at 50 Hz to keep the ESP32 watchdog alive
# send:    12 pulses (us), comma separated, '\n' at the end
# receive: "OK p0 ... p11" or "Err: ..."
# Ctrl+C stops, servos go off 200 ms later (watchdog)

import sys
import time

import serial

PORT = "/dev/ttyUSB0"
BAUD = 115200  # same as Serial.begin() in main.cpp

RATE_HZ = 50
CYCLE_TIME = 1.0 / RATE_HZ  # 20 ms

# same as HOME_US in main.cpp: shoulder, thigh, calf per leg
HOME_US = [1525, 1150, 1500,   # FR
           1650, 1700, 1500,   # FL
           1600, 1225, 1500,   # RR
           1500, 1700, 1500]   # RL


def read_line(ser):
    """one line from the ESP32, '' on timeout"""
    raw = ser.readline()
    # boot output can have non-text bytes
    return raw.decode(errors="replace").strip()


try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
except serial.SerialException as e:
    print(f"Could not open {PORT}: {e}")
    print("Plugged in? Right port (ls /dev/ttyUSB*)? PlatformIO monitor closed?")
    sys.exit(1)

try:
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

    command = ",".join(str(us) for us in HOME_US) + "\n"
    print("sending:", command.strip(), f"at {RATE_HZ} Hz, Ctrl+C to stop")

    loop_count = 0
    # start the clock here, after the boot wait, or the loop sends a burst to catch up
    time_counter = time.perf_counter()

    while True:
        # drop old lines so the next read is the reply to this command
        ser.reset_input_buffer()
        ser.write(command.encode())

        reply = read_line(ser)
        if reply == "":
            print("No reply within 0.1 s")
        elif reply.startswith("Err"):
            print("ESP32 rejected it:", reply)
        elif loop_count % RATE_HZ == 0:
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
