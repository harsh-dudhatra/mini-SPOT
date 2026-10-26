# Mini-SPOT firmware (ESP32)

The ESP32 drives the 12 servos through a PCA9685 and (soon) reads the IMU and
foot sensors. It talks to the Jetson Nano / laptop over USB serial.

Arduino framework, built with [PlatformIO](https://platformio.org/) in VS Code.

```
Jetson / laptop ──USB serial──▶ ESP32 ──I2C──▶ PCA9685 ──PWM──▶ 12 servos
                                  │
                                  ├──I2C──▶ BNO055 IMU        (planned)
                                  └──ADC──▶ 4 FSR foot sensors (planned)
```

Wiring diagram: [docs/wiring_devkit.svg](../docs/wiring_devkit.svg)

---

## Folder layout

```
firmware/
├── platformio.ini          ← build targets (see below)
├── include/board_pins.h    ← pin numbers per board, picked by a build flag
├── src/main.cpp            ← the robot firmware (servo command receiver)
└── experiments/            ← test programs, not part of the robot firmware
    ├── servo_test.cpp          first sweep test of all servos
    └── servo_test_limit.cpp    limit finder: jog one servo, read the pulse
```

## Build targets

Pick the target in the PlatformIO sidebar (or the `env:` switcher in the status bar),
then **Build → Upload → Monitor**.

| Target | Board | Builds | Use it for |
|---|---|---|---|
| `esp32dev` | ESP32 DevKit (WROOM-32U, CP2102) | `src/main.cpp` | the robot firmware |
| `esp32s2` | ESP32-S2 Saola-1 (old board) | `src/main.cpp` | old perfboard |
| `limits` | ESP32 DevKit | `experiments/servo_test_limit.cpp` | finding home + limits |
| `limits_s2` | ESP32-S2, old servo channels | `experiments/servo_test_limit.cpp` | same, on the old perfboard |

From a terminal:
```bash
pio run -e esp32dev -t upload
pio device monitor -e esp32dev
```

## Pins

| Signal | ESP32 DevKit | ESP32-S2 (old) |
|---|---|---|
| I2C SDA | GPIO21 | GPIO36 |
| I2C SCL | GPIO22 | GPIO35 |
| PCA9685 OE (LOW = servos on) | GPIO25 | GPIO7 (not wired on the old perfboard) |
| FSR FR / FL / RR / RL (planned) | GPIO32 / 33 / 34 / 35 | – |

I2C addresses: PCA9685 `0x40`, BNO055 `0x28` or `0x29`.

## Servo channels

One PCA9685 header block per leg, order shoulder, thigh, calf (channels 3, 7, 11, 15 free):

| Leg | Shoulder | Thigh | Calf |
|---|---|---|---|
| FR | 0 | 1 | 2 |
| FL | 4 | 5 | 6 |
| RR | 8 | 9 | 10 |
| RL | 12 | 13 | 14 |

All per-servo arrays in the code (`CHANNEL`, `NAME`, `HOME_US`, `MIN_US`, `MAX_US`)
use this joint order. Rewiring only changes `CHANNEL`.

---

## main.cpp: serial command format

115200 baud. Send **one line with 12 pulse values in µs**, separated by spaces or commas,
in joint order (FR shoulder, FR thigh, FR calf, FL …, RR …, RL …):

```
1525 1150 1500 1650 1700 1500 1600 1225 1500 1500 1700 1500
```

Replies:

| Reply | Meaning |
|---|---|
| `OK 1525 1150 …` | line accepted, these are the pulses now sent to the servos |
| `Err: need 12 values.` | wrong number of values, nothing moved |
| `Err: FR_thigh out of limit 950-1350` | a value is outside that servo's limits, nothing moved |

### Safety

- **Startup:** OE is held HIGH (outputs off) while the PCA9685 is set up and every
  servo gets its home pulse, then OE goes LOW. The first pulse a servo ever sees is home.
- **Whole line or nothing:** all 12 values are checked before any servo moves.
- **Per-servo limits:** `MIN_US` / `MAX_US`, measured with the limit finder.
- **Step limit:** each servo moves at most `MAX_STEP` (100 µs) per command, so a
  big jump takes several commands.

### Not done yet

- watchdog (stop the servos if the Jetson stops sending)
- check that `pwm.begin()` found the PCA9685
- BNO055 IMU readout, FSR readout
- micro-ROS (reason for the move to the classic ESP32)

---

## Limit finder (`limits` / `limits_s2`)

Moves one servo at a time so you can find its home pulse and safe limits.
Single keys, no Enter needed. **Hold a key to sweep.**

| Key | Action |
|---|---|
| `d` / `a` | +10 / −10 µs |
| `D` / `A` | +50 / −50 µs |
| `n` / `p` | next / previous servo (the current one goes home first) |
| `h` | current servo back to home |
| `e` / `x` | outputs on / off (PCA9685 OE) |

Each key prints e.g. `FR_thigh ch 1 pulse 1210 us(home) 1150, offset 60)`.

How to measure a joint (robot on a stand, legs free, hand on the main switch):
1. With the power off, move the joint by hand to find where it touches or binds. Mark it.
2. Power on, hold `d` until just before the mark, tap `a` to back off, note the pulse.
3. `h`, then the same in the other direction with `a`.
4. Note the direction too (left and right servos are mirrored).

The first version of this tool used typed commands (`30+` + Enter); it is still in the
file as a comment.

---

## LAB items (placeholders until measured)

| Item | Where | Status |
|---|---|---|
| Home pulse per servo | `HOME_US` | measured on the old board, RR leg broken, not yet in the code |
| Min / max pulse per servo | `MIN_US`, `MAX_US` | measured except RR, code still has home ±200 |
| PCA9685 oscillator | `setOscillatorFrequency(25000000)` | nominal, measure a 1500 µs pulse with a scope |
| OE pull-down on the PCA9685 board | wiring | check before adding a pull-up |
| Servo direction (SIGN) per joint | Jetson side | compare sim vs. robot |
| Coupled thigh/calf limits | – | only if the gait needs more range than the box |
