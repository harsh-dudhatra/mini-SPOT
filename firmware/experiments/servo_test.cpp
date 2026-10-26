#include <Arduino.h>
#include "board_pins.h"
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

const uint16_t SERVOMIN = 150;
const uint16_t SERVOMAX = 600;
const float SERVO_FQ = 50;

int servonum = 0;

void setup() {
  pinMode(PCA_OE_PIN, OUTPUT);
  pinMode(PCA_OE_PIN, HIGH);
  Serial.begin(115200);
  Serial.println("Servo Test");
  Wire.begin(SDA_PIN, SCL_PIN);
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FQ);
  pinMode(PCA_OE_PIN, LOW);
  delay(10);
}
void loop() {
  Serial.println("Setting up servo: " + servonum);
  for (uint16_t pulselen = SERVOMIN; pulselen < SERVOMAX; pulselen++) {
    pwm.setPWM(servonum, 0, pulselen);
  }
  delay(500);
  for (uint16_t pulselen = SERVOMAX; pulselen > SERVOMIN; pulselen--) {
    pwm.setPWM(servonum, 0, pulselen);
  }
  delay(500);
  servonum++;
  if (servonum > 11) servonum = 0;
}
