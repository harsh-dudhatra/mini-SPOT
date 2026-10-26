#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include "board_pins.h"

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

const int NUM_SERVOS = 12; // total 12 servos
// order per leg: shoulder, thigh, calf
// one PCA header block per leg (ch 3, 7, 11, 15 free)  
const int CHANNEL[NUM_SERVOS] = { 0,  1,  2,   // FR
                                  4,  5,  6,   // FL
                                  8,  9, 10,   // RR
                                 12, 13, 14};  // RL
const char* NAME[NUM_SERVOS] = {"FR_shoulder", "FR_thigh", "FR_calf",
                                "FL_shoulder", "FL_thigh", "FL_calf",
                                "RR_shoulder", "RR_thigh", "RR_calf",
                                "RL_shoulder", "RL_thigh", "RL_calf",}; //assign name to each servo i

// LAB: home pulses from the old code, same order as CHANNEL
const int HOME_US[NUM_SERVOS] = {1525, 1150, 1500,   // FR
                                 1650, 1700, 1500,   // FL
                                 1600, 1225, 1500,   // RR
                                 1500, 1700, 1500};  // RL

// LAB: home +- 200 for now, replace with values from the limit finder
const int MIN_US[NUM_SERVOS] = {1325,  950, 1300,   // FR
                                1450, 1500, 1300,   // FL
                                1400, 1025, 1300,   // RR
                                1300, 1500, 1300};  // RL
const int MAX_US[NUM_SERVOS] = {1725, 1350, 1700,   // FR
                                1850, 1900, 1700,   // FL
                                1800, 1425, 1700,   // RR
                                1700, 1900, 1700};  // RL

const int MAX_STEP = 100; // max step at a time, not letting servo jump

int pulse[NUM_SERVOS];    // pulse each servo has right now
int newPulse[NUM_SERVOS]; // pulse asked for in the last command
char line[100];


bool readUsFromCommand() {
  int len = Serial.readBytesUntil('\n', line, sizeof(line) - 1);
  line[len] = '\0';

  int count = 0;
  char* token = strtok(line, ", \r");
  while(token != NULL && count < NUM_SERVOS) {
    newPulse[count] = atoi(token);
    count++;
    token = strtok(NULL, ", \r");
  }

  if (count != NUM_SERVOS) {
    Serial.println("Err: need 12 values.");
    return false;
  }

  // check all values first, so a bad one doesn't leave half the servos moved
  for (int i = 0; i < NUM_SERVOS; i++) {
    if (newPulse[i] < MIN_US[i] || newPulse[i] > MAX_US[i]) {
      Serial.print("Err: ");
      Serial.print(NAME[i]);
      Serial.print(" out of limit ");
      Serial.print(MIN_US[i]);
      Serial.print("-");
      Serial.println(MAX_US[i]);
      return false;
    }
  }
  return true;
}

// clamp to limits, write to the servo, remember it
void setPulse(int i, int us) {
  pulse[i] = constrain(us, MIN_US[i], MAX_US[i]);
  pwm.writeMicroseconds(CHANNEL[i], pulse[i]);
}

// one short line for all servos, so it stays fast when the jetson sends often
void printPulses() {
  Serial.print("OK");
  for (int i = 0; i < NUM_SERVOS; i++) {
    Serial.print(" ");
    Serial.print(pulse[i]);
  }
  Serial.println();
}

void setup() {
  pinMode(PCA_OE_PIN, OUTPUT);
  digitalWrite(PCA_OE_PIN, HIGH);
  Serial.begin(115200);
  Serial.setTimeout(50); // don't wait 1s for a broken line
  Wire.begin(SDA_PIN, SCL_PIN);
  pwm.begin();
  pwm.setOscillatorFrequency(25000000); // 25MHz
  pwm.setPWMFreq(50);
  delay(10);

  // no step limit here, we don't know where the servos are at startup
  for (int i = 0; i < NUM_SERVOS; i++) setPulse(i, HOME_US[i]);
  delay(100);
  digitalWrite(PCA_OE_PIN, LOW);
  printPulses();
}

void loop() {
  if (!Serial.available()) return;
  if (!readUsFromCommand()) return;

  for (int i = 0; i < NUM_SERVOS; i++) {
    int step = newPulse[i] - pulse[i];
    if (step > MAX_STEP) step = MAX_STEP;     // too far up, only go MAX_STEP
    if (step < -MAX_STEP) step = -MAX_STEP;   // too far down, same
    setPulse(i, pulse[i] + step);
  }
  printPulses();
}
