#include "Drone.h"

namespace Drone {

void Drone::init() {
  _flyHandler.init();
  // initGPS();  // requiere un 3er UART — no disponible en C3 mini
  updateData();
}

void Drone::initGPS() {
  _gpsSerial.begin(9600, SERIAL_8N1, 16, 17);
  PRINT("Init GPS");
}

void Drone::onUpdate(DeltaTime dt) {
  _flyHandler.onUpdate(dt, _movement);
  updateData();
}

void Drone::updateData() {
    // GPS pendiente de implementar
}

}  // namespace Drone
