#include <Arduino.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include <WebServer.h>
#include <Wire.h>
#include <ArduinoJson.h>
#include <TinyGPSPlus.h>
#include "config.h"

// ── State ─────────────────────────────────────────────────────────────────────

static WebServer   server(SERVER_PORT);
static TinyGPSPlus gps;

static bool  imu_ok    = false;
static float roll_deg  = 0.0f;
static float pitch_deg = 0.0f;

static unsigned long last_imu_ms = 0;

// ── IMU – Wire directo (MPU-6500 / MPU-9250) ──────────────────────────────────

static uint8_t imuAddr = IMU_ADDR;

static void imuWrite(uint8_t reg, uint8_t val) {
    Wire.beginTransmission(imuAddr);
    Wire.write(reg);
    Wire.write(val);
    Wire.endTransmission();
}

static uint8_t imuWhoAmI(uint8_t addr) {
    Wire.beginTransmission(addr);
    Wire.write(0x75);
    if (Wire.endTransmission(false) != 0) return 0x00;
    Wire.requestFrom(addr, (uint8_t)1);
    if (!Wire.available()) return 0x00;
    return Wire.read();
}

static bool imuInit() {
    for (uint8_t addr : {(uint8_t)0x68, (uint8_t)0x69}) {
        uint8_t who = imuWhoAmI(addr);
        LOG("[IMU] addr=0x%02X  WHO_AM_I=0x%02X\n", addr, who);
        if (who == 0x70 || who == 0x71 || who == 0x68 || who == 0x69) {
            imuAddr = addr;
            LOG("[IMU] Detectado en 0x%02X (WHO_AM_I=0x%02X)\n", addr, who);
            goto configure;
        }
    }
    LOG("[IMU] No encontrado en 0x68 ni 0x69 — revisar SDA/SCL y alimentacion\n");
    return false;

configure:
    imuWrite(0x6B, 0x80);  // reset completo
    delay(100);
    imuWrite(0x6B, 0x01);  // wake up, clock desde gyro X
    delay(10);
    imuWrite(0x1A, 0x03);  // DLPF ≈ 41 Hz
    imuWrite(0x19, 0x03);  // SMPLRT_DIV=3 → 250 Hz
    imuWrite(0x1B, 0x00);  // GYRO_CONFIG:  ±250 °/s
    imuWrite(0x1C, 0x00);  // ACCEL_CONFIG: ±2 g
    return true;
}

static void imuRead(float &ax, float &ay, float &az,
                    float &gx, float &gy, float &gz) {
    Wire.beginTransmission(imuAddr);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    Wire.requestFrom(imuAddr, (uint8_t)14);

    auto rd = []() -> int16_t {
        return (int16_t)((Wire.read() << 8) | Wire.read());
    };

    ax = rd() * ACCEL_SCALE;
    ay = rd() * ACCEL_SCALE;
    az = rd() * ACCEL_SCALE;
    rd();                     // temperatura (ignorar)
    gx = rd() * GYRO_SCALE;
    gy = rd() * GYRO_SCALE;
    gz = rd() * GYRO_SCALE;
}

// ── HTTP handlers ─────────────────────────────────────────────────────────────

static void handle_debug() {
    JsonDocument doc;

    // I2C scan
    JsonArray devs = doc["i2c_devices"].to<JsonArray>();
    for (uint8_t addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        if (Wire.endTransmission() == 0) devs.add(addr);
    }

    // WHO_AM_I en 0x68 y 0x69
    for (uint8_t addr : {(uint8_t)0x68, (uint8_t)0x69}) {
        Wire.beginTransmission(addr);
        Wire.write(0x75);
        if (Wire.endTransmission(false) == 0) {
            Wire.requestFrom(addr, (uint8_t)1);
            if (Wire.available()) {
                char key[20];
                snprintf(key, sizeof(key), "who_am_i_0x%02X", addr);
                doc[key] = Wire.read();
            }
        }
    }

    doc["imu_ok"]   = imu_ok;
    doc["imu_addr"] = imuAddr;

    String body;
    serializeJson(doc, body);
    server.send(200, "application/json", body);
}

static void handle_telemetry() {
    bool gps_fix = gps.location.isValid() &&
                   gps.location.age() < GPS_MAX_AGE_MS;

    JsonDocument doc;

    // IMU
    doc["roll"]   = roundf(roll_deg  * 100.0f) / 100.0f;
    doc["pitch"]  = roundf(pitch_deg * 100.0f) / 100.0f;
    doc["imu_ok"] = imu_ok;

    // GPS
    doc["gps_ok"]     = gps_fix;
    doc["satellites"] = gps.satellites.isValid() ? (int)gps.satellites.value() : 0;
    doc["lat"]        = gps_fix ? gps.location.lat()    : 0.0;
    doc["lng"]        = gps_fix ? gps.location.lng()    : 0.0;
    doc["altitude_m"] = (gps_fix && gps.altitude.isValid()) ? gps.altitude.meters() : 0.0;
    doc["speed_kmh"]  = (gps_fix && gps.speed.isValid())    ? gps.speed.kmph()      : 0.0;

    String body;
    serializeJson(doc, body);
    server.send(200, "application/json", body);
}

// ── Setup ─────────────────────────────────────────────────────────────────────

void setup() {
    Serial.begin(115200);
    delay(400);
    LOG("\n[turbodrone esp32-c3] boot\n");

    // IMU — esperar 200ms para que el MPU tenga tiempo de estabilizarse tras el reset
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(400000);
    delay(200);

    // ── I2C scanner (debug) ───────────────────────────────────────────────────
    LOG("[I2C] Escaneando bus (SDA=GPIO%d  SCL=GPIO%d)...\n", I2C_SDA, I2C_SCL);
    bool found_any = false;
    for (uint8_t addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        if (Wire.endTransmission() == 0) {
            LOG("[I2C] Dispositivo en 0x%02X\n", addr);
            found_any = true;
        }
    }
    if (!found_any) LOG("[I2C] Ningun dispositivo respondio — revisar SDA/SCL/VCC\n");

    imu_ok = imuInit();
    if (!imu_ok) LOG("[IMU] fallo — revisar cableado y IMU_ADDR en config.h");

    if (imu_ok) {
        float ax, ay, az, gx, gy, gz;
        imuRead(ax, ay, az, gx, gy, gz);
        roll_deg  = atan2f(ay, az)                    * RAD_TO_DEG;
        pitch_deg = atan2f(-ax, sqrtf(ay*ay + az*az)) * RAD_TO_DEG;
    }

    // GPS
    Serial1.begin(GPS_BAUD, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
    LOG("[GPS] UART1 iniciado — RX=GPIO%d TX=GPIO%d @%d baud\n",
        GPS_RX_PIN, GPS_TX_PIN, GPS_BAUD);

    // WiFi
    LOG("[WiFi] conectando a %s", WIFI_SSID);
    WiFi.mode(WIFI_STA);
#ifdef STATIC_IP
    {
        IPAddress ip(STATIC_IP), gw(STATIC_GATEWAY), sn(STATIC_SUBNET), dns(STATIC_DNS);
        WiFi.config(ip, gw, sn, dns);
    }
#endif
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) { delay(300); LOG("."); }
    String ip = WiFi.localIP().toString();
    LOG("\n[WiFi] IP: %s\n", ip.c_str());

    if (MDNS.begin(MDNS_HOSTNAME)) {
        MDNS.addService("http", "tcp", SERVER_PORT);
        LOG("[mDNS] http://%s.local/telemetry\n", MDNS_HOSTNAME);
    } else {
        LOG("[HTTP] fallback → http://%s/telemetry\n", ip.c_str());
    }

    server.on("/telemetry", HTTP_GET, handle_telemetry);
    server.on("/debug",    HTTP_GET, handle_debug);
    server.onNotFound([]() { server.send(404, "text/plain", "not found"); });
    server.begin();
    LOG("[HTTP] servidor iniciado\n");
}

// ── Loop ──────────────────────────────────────────────────────────────────────

void loop() {
    // GPS: drenar UART antes de cualquier otra cosa para no perder bytes
    while (Serial1.available()) {
        gps.encode(Serial1.read());
    }

    server.handleClient();

    unsigned long now = millis();
    if (now - last_imu_ms < (unsigned long)IMU_LOOP_MS) return;
    last_imu_ms = now;

    if (!imu_ok) return;

    float ax, ay, az, gx, gy, gz;
    imuRead(ax, ay, az, gx, gy, gz);

    float dt = IMU_LOOP_MS / 1000.0f;

    float accel_roll  = atan2f(ay, az)                    * RAD_TO_DEG;
    float accel_pitch = atan2f(-ax, sqrtf(ay*ay + az*az)) * RAD_TO_DEG;

    roll_deg  = ALPHA * (roll_deg  + gx * dt) + (1.0f - ALPHA) * accel_roll;
    pitch_deg = ALPHA * (pitch_deg + gy * dt) + (1.0f - ALPHA) * accel_pitch;

    static uint8_t tick = 0;
    if (++tick >= 100) {
        tick = 0;
        bool fix = gps.location.isValid() && gps.location.age() < GPS_MAX_AGE_MS;
        LOG("[IMU] roll=%.1f°  pitch=%.1f°  |  [GPS] fix=%s  sat=%d  lat=%.6f  lng=%.6f\n",
            roll_deg, pitch_deg,
            fix ? "SI" : "NO",
            gps.satellites.isValid() ? (int)gps.satellites.value() : 0,
            fix ? gps.location.lat() : 0.0,
            fix ? gps.location.lng() : 0.0);
    }
}
