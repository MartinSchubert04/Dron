#pragma once

#include <Arduino.h>
#include <Wire.h>
#include "Vec3.h"
#include "DeltaTime.h"
#include "Motor.h"
#include "PID.h"
#include "Controller.h"
#include "pch.h"

namespace Drone {

struct IMUData {
    Vec3 acc;
    Vec3 gyro;
};

class FlyHandler {
public:
    Motor motorFL, motorFR, motorBL, motorBR;

    PID pidRoll {1.2f, 0.01f, 0.4f};
    PID pidPitch{1.2f, 0.01f, 0.4f};
    PID pidYaw  {2.0f, 0.0f,  0.0f};

    FlyHandler() = default;

    void init();
    void onUpdate(DeltaTime dt, Movement& mov);

    float getRoll()  const { return _roll;  }
    float getPitch() const { return _pitch; }

private:
    void    initIMU();
    void    calibrateGyro();
    bool    beginRead();
    IMUData readIMU();

    uint8_t _address  = 0x68;
    bool    _imuReady = false;

    float _alpha = 0.98f;
    float _roll  = 0;
    float _pitch = 0;

    Vec3 _gyroBias{};
};

}  // namespace Drone
