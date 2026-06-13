#pragma once

// ── Debug ─────────────────────────────────────────────────────────────────────
#define LOG(...) Serial.printf(__VA_ARGS__)

// ── WiFi ─────────────────────────────────────────────────────────────────────
#define WIFI_SSID "MOVISTAR-2.4"
#define WIFI_PASSWORD "Martin2004"
#define MDNS_HOSTNAME "drone"

// IP estática — comentar las 4 líneas para usar DHCP
#define STATIC_IP 192, 168, 2, 150
#define STATIC_GATEWAY 192, 168, 2, 1
#define STATIC_SUBNET 255, 255, 255, 0
#define STATIC_DNS 192, 168, 2, 1

// ── HTTP server ───────────────────────────────────────────────────────────────
// Tras conectarse el ESP32 imprime su IP por Serial.
// Configurar en el backend:  ESP32_TELEMETRY_URL=http://<IP>/telemetry
// O desde el frontend en Config → ESP32-C3 (TELEMETRÍA IMU)
#define SERVER_PORT 80

// ── I2C / MPU-6500 ───────────────────────────────────────────────────────────
#define I2C_SDA 2
#define I2C_SCL 3
#define IMU_ADDR 0x68                  // AD0=GND → 0x68  |  AD0=VCC → 0x69
#define ACCEL_SCALE (1.0f / 16384.0f)  // ±2 g  → LSB a g
#define GYRO_SCALE (1.0f / 131.0f)     // ±250 °/s → LSB a °/s

// ── Filtro complementario ────────────────────────────────────────────────────
#define ALPHA 0.98f

// ── Frecuencia de lectura IMU ────────────────────────────────────────────────
#define IMU_LOOP_MS 10  // 100 Hz

// ── GPS (NEO-6M / GY-NEO6MV2 via UART1) ─────────────────────────────────────
// Cableado: GPS TX → GPIO4, GPS RX → GPIO5, VCC → 3.3V, GND → GND
#define GPS_RX_PIN 4
#define GPS_TX_PIN 5
#define GPS_BAUD 9600
// Umbral de antigüedad: si el último fix tiene más de este tiempo (ms) se marca sin fix
#define GPS_MAX_AGE_MS 3000
