#pragma once

#include <Arduino.h>
#include <stdint.h>

#define DEBUG
#ifdef DEBUG
  #define PRINT(...) Serial.printf(__VA_ARGS__)
#else
  #define PRINT(...)
#endif

// Motores PWM — IN1 de cada canal DRV8833
#define MOTOR_PIN1  2   // FL
#define MOTOR_PIN2  3   // FR
#define MOTOR_PIN3  4   // BL
#define MOTOR_PIN4  5   // BR

// MPU6500 I2C
#define MPU_SDA_PIN  6
#define MPU_SCL_PIN  7

// Joystick
#define DEADZONE  20
