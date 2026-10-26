// Servo jogs one step at a time from serial monitor
// Single keys, no Enter needed (hold a key to sweep):
//   d / a  -> +10 / -10 us      D / A  -> +50 / -50 us
//   n / p  -> next / previous servo (current goes home first)
//   h      -> home      e -> outputs on      x -> outputs off

#include<Arduino.h>
#include<Wire.h>
#include<Adafruit_PWMServoDriver.h>
#include "board_pins.h"

const int NUM_SERVOS = 12; // total 12 servos
// order per leg: shoulder, thigh, calf
#if defined(OLD_WIRING)
// old S2 perfboard, servos still plugged in the old channels
const int CHANNEL[NUM_SERVOS] = {12, 14, 13,   // FR
                                  8, 10,  9,   // FL
                                  0,  1,  2,   // RR
                                  4,  6,  5};  // RL
#else
// one PCA header block per leg (ch 3, 7, 11, 15 free)
const int CHANNEL[NUM_SERVOS] = { 0,  1,  2,   // FR
                                  4,  5,  6,   // FL
                                  8,  9, 10,   // RR
                                 12, 13, 14};  // RL
#endif
const char* NAME[NUM_SERVOS] = {"FR_shoulder", "FR_thigh", "FR_calf",
                                "FL_shoulder", "FL_thigh", "FL_calf",
                                "RR_shoulder", "RR_thigh", "RR_calf",
                                "RL_shoulder", "RL_thigh", "RL_calf",}; //assign name to each servo i

// LAB: home pulses from the old code, same order as CHANNEL
const int HOME_US[NUM_SERVOS] = {1525, 1150, 1500,   // FR
                                 1650, 1700, 1500,   // FL
                                 1600, 1225, 1500,   // RR
                                 1500, 1700, 1500};  // RL

const int ABS_MAX_US = 3000; // absolute max limit
const int ABS_MIN_US = 0; // absolute min limit
const int MAX_STEP = 300; // max step at a time, not letting servo jump
const int SMALL_STEP = 10; // a / d
const int BIG_STEP = 50;   // A / D

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

int pulse[NUM_SERVOS];
int current = 0;
int lastStep = 10;

void setPulse(int i, int us) {
  pulse[i] = constrain(us, ABS_MIN_US, ABS_MAX_US);
  pwm.writeMicroseconds(CHANNEL[i], pulse[i]);
}

void printStatus() {
  Serial.print(NAME[current]);
  Serial.print(" ch ");
  Serial.print(CHANNEL[current]);
  Serial.print(" pulse ");
  Serial.print(pulse[current]);
  Serial.print(" us(home) ");
  Serial.print(HOME_US[current]);
  Serial.print(", offset ");
  Serial.print(pulse[current] - HOME_US[current]);
  Serial.println(")");
}

void setup() {
  pinMode(PCA_OE_PIN, OUTPUT);
  digitalWrite(PCA_OE_PIN, HIGH); // all output off until user enable it (e)
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  pwm.begin();
  pwm.setOscillatorFrequency(25000000); // 25MHz
  pwm.setPWMFreq(50); // 50Hz
  delay(10);

  for (int i = 0; i < NUM_SERVOS; i++) setPulse(i, HOME_US[i]);
  Serial.println("Servo limit finder, Outputs are disabled, enter e to enable.");
  printStatus();
}

// new: one key = one command, no Enter
void loop() {
  if (!Serial.available()) return;
  char c = Serial.read(); // read one key

  if (c == 'e') { // enable outputs
    digitalWrite(PCA_OE_PIN, LOW);
    Serial.println("Outputs enabled");
  } else if (c == 'x') { // disable output
    digitalWrite(PCA_OE_PIN, HIGH);
    Serial.println("Outputs disabled");
  } else if (c == 'h') { // back to home pos
    setPulse(current, HOME_US[current]);
  } else if (c == 'n') { // next channel
    setPulse(current, HOME_US[current]);
    current = (current + 1) % NUM_SERVOS; // 11 wraps to 0
  } else if (c == 'p') { // previous channel
    setPulse(current, HOME_US[current]);
    current = (current + NUM_SERVOS - 1) % NUM_SERVOS; // 0 wraps to 11
  } else if (c == 'd') {
    setPulse(current, pulse[current] + SMALL_STEP);
  } else if (c == 'a') {
    setPulse(current, pulse[current] - SMALL_STEP);
  } else if (c == 'D') {
    setPulse(current, pulse[current] + BIG_STEP);
  } else if (c == 'A') {
    setPulse(current, pulse[current] - BIG_STEP);
  } else {
    return; // any other key (also Enter) is ignored
  }
  printStatus();
}

/* old: typed commands like "30+" with Enter
void loop() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if(cmd.length()==0) return;
  char last = cmd.charAt(cmd.length() - 1);

  if (cmd == "e") { // enable outputs
    digitalWrite(PCA_OE_PIN, LOW);
    Serial.println("Outputs enabled");
  } else if (cmd == "x") { // disable output
    digitalWrite(PCA_OE_PIN, HIGH);
    Serial.println("Outputs disabled");
  } else if (cmd == "h") { // back to home pos
    setPulse(current, HOME_US[current]);
  } else if (cmd == "n") { // next channel
    setPulse(current, HOME_US[current]); // back to home pos of current channel
    current = (current + 1) % NUM_SERVOS; // then increment current channel by 1, and 11 ->  wraps to 0
  } else if (cmd == "p") {
    setPulse(current, HOME_US[current]);
    current = (current + NUM_SERVOS - 1) % NUM_SERVOS; // 0 wraps to 11
  } else if (last == '+' || last == '-') { // "30+", "30 -", or just "+"
    String number = cmd.substring(0, cmd.length()-1); // 50x -> 50
    number.trim();
    int step = lastStep;
    if (number.length() > 0) step = number.toInt();
    if (step<=0 || step > MAX_STEP) {
      Serial.println("Step must be 1-100us");
      return;
    }
    lastStep = step;
    if (last == '-') step = -step;
    setPulse(current, pulse[current] + step);
  } else {
    Serial.println("Unknown command.");
    return;
  }
  printStatus();
}
*/
