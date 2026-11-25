#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BNO055.h>
#include "board_pins.h"

// BNO055 shares the I2C bus with the PCA9685
// address 0x28 if ADR pin is LOW, 0x29 if HIGH -> try the other one if begin() fails
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x28, &Wire);

const unsigned long PRINT_MS = 50; // 20 Hz
unsigned long lastPrint = 0;       // millis() of the last print

void setup() {
  Serial.begin(115200);
  Serial.println("IMU Test");
  Wire.begin(SDA_PIN, SCL_PIN);

  // IMUPLUS = accel + gyro only, no magnetometer (servo magnets would mess it up)
  if (!bno.begin(OPERATION_MODE_IMUPLUS)) {
    Serial.println("Err: BNO055 not found, check wiring / address");
    while (true) delay(1000); // nothing to do without the IMU
  }
  bno.setExtCrystalUse(true); // breakout has a 32kHz crystal, more stable
  delay(100);
}

void loop() {
  // millis() timer instead of delay(), same way it has to work in main.cpp later
  if (millis() - lastPrint < PRINT_MS) return;
  lastPrint = millis();

  // x = heading, y = roll, z = pitch, all in degrees
  imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);

  // 0 = not calibrated, 3 = fully calibrated
  uint8_t sys, gyro, accel, mag;
  bno.getCalibration(&sys, &gyro, &accel, &mag);

  Serial.print("IMU ");
  Serial.print(euler.y(), 1); // roll
  Serial.print(" ");
  Serial.print(euler.z(), 1); // pitch
  Serial.print(" ");
  Serial.print(euler.x(), 1); // heading
  Serial.print("  cal sys/gyro/accel: ");
  Serial.print(sys);
  Serial.print("/");
  Serial.print(gyro);
  Serial.print("/");
  Serial.println(accel);
}
