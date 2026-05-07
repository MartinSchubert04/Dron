#include <Arduino.h>
#include <Bluepad32.h>
#include "pch.h"
#include "Drone.h"
#include "DeltaTime.h"

static Drone::Drone   drone;
static GamepadPtr     gamepad = nullptr;

void onConnect(GamepadPtr gp)    { gamepad = gp; }
void onDisconnect(GamepadPtr gp) { gamepad = nullptr; }

static Drone::Movement joystickToMovement(GamepadPtr gp) {
    Drone::Movement mov{};

    int ly = gp->axisY();   // left  stick Y  → throttle
    int lx = gp->axisX();   // left  stick X  → yaw
    int rx = gp->axisRX();  // right stick X  → roll
    int ry = gp->axisRY();  // right stick Y  → pitch

    // Throttle: palanca izquierda arriba (ly = -512) → máximo
    if (-ly > DEADZONE)
        mov.throttle = (int16_t)map(-ly, DEADZONE, 512, 0, 255);

    if (abs(rx) > DEADZONE) mov.roll  =  rx;
    if (abs(ry) > DEADZONE) mov.pitch = -ry;  // invertir: arriba = pitch forward
    if (abs(lx) > DEADZONE) mov.yaw   =  lx;

    return mov;
}

static uint32_t lastMs = 0;

void setup() {
#ifdef DEBUG
    Serial.begin(115200);
#endif
    BP32.setup(&onConnect, &onDisconnect);
    BP32.forgetBluetoothKeys();

    // init() calibra el giroscopio (~1 segundo, mantener el dron quieto)
    drone.init();
    lastMs = millis();
}

void loop() {
    uint32_t now = millis();
    DeltaTime dt(now - lastMs);
    lastMs = now;

    BP32.update();

    if (gamepad && gamepad->isConnected()) {
        Drone::Movement mov = joystickToMovement(gamepad);
        drone.setMovement(mov);
    }

    drone.onUpdate(dt);
}
