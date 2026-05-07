#pragma once

#include "pch.h"
namespace Drone {

struct Movement {
    int16_t throttle = 0;
    int16_t pitch    = 0;
    int16_t roll     = 0;
    int16_t yaw      = 0;
};

}  // namespace Drone
