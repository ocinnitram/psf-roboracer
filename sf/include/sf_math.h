#pragma once

#include <cmath>

namespace sf {

constexpr double kPi = 3.14159265358979323846;
constexpr double kHalfPi = 0.5 * kPi;

// cos and sin of one angle. The build keeps GCC from fusing or merging libm
// trig calls (CMakeLists.txt), so code that needs both, or needs them more
// than once, computes a rotation once and passes it on.
struct Rotation {
    double c;
    double s;
};

inline Rotation rotation(double angle_rad) {
    return {std::cos(angle_rad), std::sin(angle_rad)};
}

} // namespace sf
