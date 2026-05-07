#pragma once

// #include <TinyGPS++.h>
#include <HardwareSerial.h>
#include "Controller.h"
#include "FlyHandler.h"
#include "pch.h"

namespace Drone {

struct DroneData {
  float altitude;
  float lat;
  float lng;
  float speed;
};

class Drone {
public:
  // HardwareSerial(1) = UART1, disponible en C3 (UART0 lo usa Serial/USB)
  Drone() : _gpsSerial(1), _data({0.0f, 0.0f, 0.0f, 0.0f}) {}

  void init();

  void initGPS();

  void onUpdate(DeltaTime dt);

  void updateData();

  const DroneData &getDroneData() const { return _data; }
  const Movement &getMovement() const { return _movement; }
  const FlyHandler &getFlyHandler() const { return _flyHandler; }
  void setMovement(Movement &mov) { _movement = mov; }

private:
  DroneData _data;
  FlyHandler _flyHandler;
  Movement _movement;
  // TinyGPSPlus _gps;
  HardwareSerial _gpsSerial;
};

}  // namespace Drone
