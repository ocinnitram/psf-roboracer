// World-frame corners of a body-aligned rectangle. Length runs along yaw.

#include "sf_rect.h"

#include <array>
#include <cmath>

namespace sf {

std::array<Point2D, 4> corners(const Rect& rect) {
    const double hl = 0.5 * rect.length;
    const double hw = 0.5 * rect.width;
    const double c = std::cos(rect.yaw);
    const double s = std::sin(rect.yaw);
    const Point2D local[4] = {{hl, hw}, {hl, -hw}, {-hl, -hw}, {-hl, hw}};
    std::array<Point2D, 4> out{};
    for (int i = 0; i < 4; ++i) {
        out[i].x = local[i].x * c - local[i].y * s + rect.cx;
        out[i].y = local[i].x * s + local[i].y * c + rect.cy;
    }
    return out;
}

} // namespace sf
