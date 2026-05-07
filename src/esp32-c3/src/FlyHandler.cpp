#include "FlyHandler.h"

namespace Drone {

void FlyHandler::init() {
    motorFL.init(MOTOR_PIN1, 0);
    motorFR.init(MOTOR_PIN2, 1);
    motorBL.init(MOTOR_PIN3, 2);
    motorBR.init(MOTOR_PIN4, 3);
    initIMU();
    calibrateGyro();
}

void FlyHandler::initIMU() {
    Wire.begin(MPU_SDA_PIN, MPU_SCL_PIN);

    Wire.beginTransmission(_address);
    Wire.write(0x6B);
    Wire.write(0x00);  // wake up
    Wire.endTransmission();

    Wire.beginTransmission(_address);
    Wire.write(0x1B);
    Wire.write(0x00);  // gyro ±250°/s → 131 LSB/(°/s)
    Wire.endTransmission();

    Wire.beginTransmission(_address);
    Wire.write(0x1C);
    Wire.write(0x00);  // accel ±2g → 16384 LSB/g
    Wire.endTransmission();

    delay(100);
    _imuReady = true;
}

void FlyHandler::calibrateGyro() {
    const int N = 500;
    Vec3 sum{};
    for (int i = 0; i < N; i++) {
        Wire.beginTransmission(_address);
        Wire.write(0x43);
        Wire.endTransmission(false);
        Wire.requestFrom(_address, (uint8_t)6, (uint8_t) true);
        sum += Vec3(
            (int16_t)(Wire.read() << 8 | Wire.read()) / 131.0f,
            (int16_t)(Wire.read() << 8 | Wire.read()) / 131.0f,
            (int16_t)(Wire.read() << 8 | Wire.read()) / 131.0f
        );
        delay(2);
    }
    _gyroBias = sum / (float)N;
}

bool FlyHandler::beginRead() {
    Wire.beginTransmission(_address);
    Wire.write(0x3B);
    Wire.endTransmission(false);
    return Wire.requestFrom(_address, (uint8_t)14, (uint8_t) true) == 14;
}

IMUData FlyHandler::readIMU() {
    if (!_imuReady || !beginRead())
        return IMUData{};

    int16_t ax = Wire.read() << 8 | Wire.read();
    int16_t ay = Wire.read() << 8 | Wire.read();
    int16_t az = Wire.read() << 8 | Wire.read();
    Wire.read(); Wire.read();  // temp — descartar
    int16_t gx = Wire.read() << 8 | Wire.read();
    int16_t gy = Wire.read() << 8 | Wire.read();
    int16_t gz = Wire.read() << 8 | Wire.read();

    IMUData data;
    data.acc  = Vec3(ax, ay, az) / 16384.0f;
    data.gyro = Vec3(gx, gy, gz) / 131.0f - _gyroBias;
    return data;
}

void FlyHandler::onUpdate(DeltaTime dt, Movement& mov) {
    IMUData imu = readIMU();
    float dtSec = (float)dt;

    // Con throttle bajo, apagar motores y resetear integradores
    if (mov.throttle < 10) {
        pidRoll.reset();
        pidPitch.reset();
        pidYaw.reset();
        motorFL.setSpeed(0); motorFR.setSpeed(0);
        motorBL.setSpeed(0); motorBR.setSpeed(0);
        motorFL.onUpdate();  motorFR.onUpdate();
        motorBL.onUpdate();  motorBR.onUpdate();
        return;
    }

    // Filtro complementario roll/pitch (modo ángulo)
    float roll_acc  = atan2(imu.acc.y, imu.acc.z) * RAD_TO_DEG;
    float pitch_acc = atan2(-imu.acc.x,
                            sqrt(imu.acc.y * imu.acc.y + imu.acc.z * imu.acc.z)) * RAD_TO_DEG;

    _roll  = _alpha * (_roll  + imu.gyro.x * dtSec) + (1.0f - _alpha) * roll_acc;
    _pitch = _alpha * (_pitch + imu.gyro.y * dtSec) + (1.0f - _alpha) * pitch_acc;

    // Setpoints: joystick ±512 → ±76.8° (roll/pitch) o ±150°/s (yaw rate)
    float rollOut  = pidRoll.compute(mov.roll  * 0.15f,          _roll,        dt);
    float pitchOut = pidPitch.compute(mov.pitch * 0.15f,         _pitch,       dt);
    float yawOut   = pidYaw.compute(mov.yaw   * (150.0f / 512.f), imu.gyro.z,  dt);

    // Mixing X-config: FL=CCW, FR=CW, BL=CW, BR=CCW
    int fl = (int)(mov.throttle + rollOut - pitchOut + yawOut);
    int fr = (int)(mov.throttle - rollOut - pitchOut - yawOut);
    int bl = (int)(mov.throttle + rollOut + pitchOut - yawOut);
    int br = (int)(mov.throttle - rollOut + pitchOut + yawOut);

    motorFL.setSpeed(constrain(fl, 0, 255));
    motorFR.setSpeed(constrain(fr, 0, 255));
    motorBL.setSpeed(constrain(bl, 0, 255));
    motorBR.setSpeed(constrain(br, 0, 255));

    motorFL.onUpdate();
    motorFR.onUpdate();
    motorBL.onUpdate();
    motorBR.onUpdate();
}

}  // namespace Drone
