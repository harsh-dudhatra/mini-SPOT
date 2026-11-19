"""Xbox controller -> (vx, vy, wz) command, read straight from the Linux joystick device (no extra packages).

    left stick  up/down     forward / backward
    left stick  left/right  strafe left / right
    right stick left/right  turn left / right
    B                       stop (stand)
    Back / View             quit (same as Ctrl+C: stand, then stop)

Works with wired, wireless-adapter and Bluetooth Xbox controllers (xpad / xpadneo drivers -> /dev/input/js0).
Test it alone:  python3 -m sim.gamepad
Docker: /dev/input must be shared with the container (see docker/docker-compose.yml).

By default only the stick axis pushed furthest is used (one movement at a time); --gamepad-mix allows
combinations. A pushed stick maps linearly onto the training range of its axis in CMD_RANGES
(e.g. vx 0 .. 0.20 m/s forward, 0 .. -0.18 m/s backward), never to speeds the policy has not seen.
"""
import os
import struct
import sys
import time

import numpy as np

from sim.command_env import CMD_RANGES

JS_EVENT = struct.Struct("IhBB")          # time [ms], value, type, number
JS_BUTTON, JS_AXIS, JS_INIT = 0x01, 0x02, 0x80

# xpad / xpadneo axis and button numbers
AX_LX, AX_LY, AX_RX = 0, 1, 3
BTN_B, BTN_BACK = 1, 6
DEADZONE = 0.2


class QuitRequested(Exception):
    pass


class Gamepad:
    def __init__(self, device="/dev/input/js0", mix=False):
        self.device = device
        self.mix = mix
        self.fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
        self.axes = {}
        self.last = None

    def _poll(self):
        """Read all waiting events. Returns the set of buttons pressed since the last poll."""
        pressed = set()
        while True:
            try:
                data = os.read(self.fd, JS_EVENT.size * 32)
            except BlockingIOError:
                break
            except OSError:
                raise QuitRequested("controller disconnected")
            if not data:
                break
            for i in range(0, len(data) - JS_EVENT.size + 1, JS_EVENT.size):
                _, value, etype, num = JS_EVENT.unpack_from(data, i)
                if etype & JS_AXIS:
                    self.axes[num] = value / 32767.0
                elif etype & JS_BUTTON and value and not etype & JS_INIT:
                    pressed.add(num)
        return pressed

    @staticmethod
    def _scale(s, rng):
        """Stick value in [-1, 1] -> speed in the (low, high) training range of that axis."""
        if abs(s) < DEADZONE:
            return 0.0
        k = (abs(s) - DEADZONE) / (1.0 - DEADZONE)
        return float(k * (rng[1] if s > 0 else rng[0]))

    def read_command(self):
        """(vx, vy, wz) when the command changed, else None. Raises QuitRequested on Back."""
        pressed = self._poll()
        if BTN_BACK in pressed:
            raise QuitRequested("Back pressed")
        if BTN_B in pressed:
            self.axes = {}
        # stick up = negative axis value
        sticks = np.array([-self.axes.get(AX_LY, 0.0), -self.axes.get(AX_LX, 0.0), -self.axes.get(AX_RX, 0.0)])
        cmd = np.array([
            self._scale(sticks[0], CMD_RANGES["vx"]),
            self._scale(sticks[1], CMD_RANGES["vy"]),
            self._scale(sticks[2], CMD_RANGES["wz"]),
        ])
        if not self.mix and np.any(cmd):
            keep = np.argmax(np.abs(sticks) * (cmd != 0))
            cmd = np.where(np.arange(3) == keep, cmd, 0.0)
        cmd = np.round(cmd, 2)                       # ignore tiny stick jitter
        if self.last is not None and np.array_equal(cmd, self.last):
            return None
        self.last = cmd
        return cmd

    def close(self):
        os.close(self.fd)


def open_gamepad(device, mix=False):
    try:
        pad = Gamepad(device, mix)
    except OSError as e:
        sys.exit(f"cannot open {device}: {e}\n  is the controller connected (ls /dev/input/js*)? "
                 f"in Docker, is /dev/input shared?")
    print(f">> gamepad {device}: left stick move/strafe, right stick turn, B stop, Back quit")
    return pad


if __name__ == "__main__":
    pad = open_gamepad(sys.argv[1] if len(sys.argv) > 1 else "/dev/input/js0")
    try:
        while True:
            cmd = pad.read_command()
            if cmd is not None:
                print(f"(vx, vy, wz) = ({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f})")
            time.sleep(0.02)
    except (KeyboardInterrupt, QuitRequested) as e:
        print(f"\n{e or 'Ctrl+C'}")
