#pragma once

// ── Debug ─────────────────────────────────────────────────────────────────────
#define LOG(...) Serial.printf(__VA_ARGS__)

// ── WiFi ─────────────────────────────────────────────────────────────────────
#define WIFI_SSID "MOVISTAR-2.4"
#define WIFI_PASSWORD "Martin2004"
#define MDNS_HOSTNAME "drone"

// ── HTTP server ───────────────────────────────────────────────────────────────
#define SERVER_PORT 80

// ── Telemetría UDP (push ESP32 → backend) ────────────────────────────────────
// El ESP32 hace broadcast de JSON al puerto TELEMETRY_UDP_PORT.
// El backend escucha en 0.0.0.0:TELEMETRY_UDP_PORT — no requiere configurar URL.
#define TELEMETRY_UDP_PORT 4210
#define TELEMETRY_HZ       20   // 20 paquetes/s → 50 ms entre envíos

// ── I2C / MPU-6500 ───────────────────────────────────────────────────────────
#define I2C_SDA 6
#define I2C_SCL 7
#define IMU_ADDR     0x68              // AD0=GND → 0x68  |  AD0=VCC → 0x69
#define ACCEL_SCALE  (1.0f / 16384.0f) // ±2 g  → LSB a g
#define GYRO_SCALE   (1.0f / 131.0f)   // ±250 °/s → LSB a °/s

// ── Filtro complementario ────────────────────────────────────────────────────
#define ALPHA 0.98f

// ── Frecuencia de lectura IMU ────────────────────────────────────────────────
#define IMU_LOOP_MS 10  // 100 Hz

// ── GPS (NEO-6M / GY-NEO6MV2 via UART1) ─────────────────────────────────────
// Cableado: GPS TX → GPIO4, GPS RX → GPIO5, VCC → 3.3V, GND → GND
#define GPS_RX_PIN  4
#define GPS_TX_PIN  5
#define GPS_BAUD    9600
// Umbral de antigüedad: si el último fix tiene más de este tiempo (ms) se marca sin fix
#define GPS_MAX_AGE_MS 3000
