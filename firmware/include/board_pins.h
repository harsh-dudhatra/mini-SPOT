#pragma once

// Pin map for every board we build for. The board is picked in platformio.ini
// with a build flag (-D BOARD_ESP32S2 or -D BOARD_ESP32DEV).

#if defined(BOARD_ESP32S2)
  // ESP32-S2 Saola-1 (current board)
  const int SDA_PIN    = 36;  // I2C data  -> PCA9685 SDA
  const int SCL_PIN    = 35;  // I2C clock -> PCA9685 SCL
  const int PCA_OE_PIN = 7;   // PCA9685 Output Enable: LOW = servos on, HIGH = servos off

#elif defined(BOARD_ESP32DEV)
  // Classic ESP32 DevKit (neded for micro-ROS)
  // GPIO 34-39 are input-only on this chip, so the S2's I2C pins can't be reused.
  const int SDA_PIN    = 21;  // I2C data  -> default I2C pin on the DevKit
  const int SCL_PIN    = 22;  // I2C clock -> default I2C pin on the DevKit
  const int PCA_OE_PIN = 25;  // PCA9685 Output Enable: LOW = servos on, HIGH = servos off

// #else
//   #error "No board selected"
#endif
