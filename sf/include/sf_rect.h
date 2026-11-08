#pragma once

#include "track.h"

#include <array>

// World-frame rectangle with length along yaw and optional tracked velocity.

namespace sf {

struct Rect {
    double cx = 0.0;
    double cy = 0.0;
    double yaw = 0.0;
    double length = 0.0;
    double width = 0.0;
    double vx = 0.0;
    double vy = 0.0;
};

std::array<Point2D, 4> corners(const Rect& rect);

} // namespace sf
