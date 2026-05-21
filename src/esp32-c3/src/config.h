#pragma once

// ── WiFi ──────────────────────────────────────────────────────────────────────
#define WIFI_SSID    "YOUR_WIFI_SSID"
#define WIFI_PASS    "YOUR_WIFI_PASSWORD"

// ── WebSocket server ──────────────────────────────────────────────────────────
#define WS_PORT      81

// ── Motor PWM pins (DRV8833 AIN1/BIN1 — AIN2/BIN2 tied to GND) ───────────────
#define PIN_MOTOR_FL  2   // Front-left
#define PIN_MOTOR_FR  3   // Front-right
#define PIN_MOTOR_BL  4   // Back-left
#define PIN_MOTOR_BR  5   // Back-right

// ── PWM ───────────────────────────────────────────────────────────────────────
#define PWM_FREQ     20000   // Hz
#define PWM_BITS     8       // resolution bits (0-255)
#define PWM_MAX      255

// ── Safety ────────────────────────────────────────────────────────────────────
#define WATCHDOG_MS  500     // disarm if no command received in this many ms

// ── Status broadcast ──────────────────────────────────────────────────────────
#define STATUS_MS    100     // send motor values to app every N ms

// ── Debug ─────────────────────────────────────────────────────────────────────
#define DEBUG
#ifdef DEBUG
  #define LOG(...) Serial.printf(__VA_ARGS__)
#else
  #define LOG(...)
#endif
