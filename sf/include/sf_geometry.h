#pragma once

#include "vehicle/params.h"
#include "sf_math.h"
#include "sf_model.h"
#include "sf_rect.h"
#include "sf_tuning.h"

#include <algorithm>
#include <cmath>

// Body cover geometry and rectangle distances. Header-only.

namespace sf {

struct Point {
    double x;
    double y;
};

// Cover disks along the body axis: `cover_disks` slices of the rectangle,
// each covered by the circumcircle of its slice. Index 0 is forward-most.
inline double cover_disk_radius() {
    return 0.5 *
           std::hypot(kEgoLengthM / SafetyParameters::cover_disks, kEgoWidthM);
}

inline double cover_disk_offset(int index) {
    const double spacing = kEgoLengthM / SafetyParameters::cover_disks;
    return 0.5 * kEgoLengthM - spacing * (static_cast<double>(index) + 0.5);
}

// World position of a body-frame point (lx forward, ly left); `heading` is
// rotation(z.psi).
inline Point body_point(const State& z, const Rotation& heading, double lx,
                        double ly) {
    const double c = heading.c;
    const double s = heading.s;
    return {z.x + lx * c - ly * s, z.y + lx * s + ly * c};
}

inline Point cover_disk_centre(const State& z, const Rotation& heading,
                               int index) {
    return body_point(z, heading, cover_disk_offset(index), 0.0);
}

// Signed distance from a point to a rectangle: positive outside, negative
// (penetration) inside. Rect local x runs along `yaw` and spans +-length/2;
// `yaw` is rotation(r.yaw).
inline double rect_signed_distance(const Rect& r, const Rotation& yaw,
                                   double px, double py) {
    const double dx = px - r.cx;
    const double dy = py - r.cy;
    const double c = yaw.c;
    const double s = yaw.s;
    const double qx = std::abs(dx * c + dy * s) - 0.5 * r.length;
    const double qy = std::abs(-dx * s + dy * c) - 0.5 * r.width;
    if (qx > 0.0 || qy > 0.0) {
        return std::hypot(std::max(qx, 0.0), std::max(qy, 0.0));
    }
    return std::max(qx, qy);
}

// Half-extent of a rectangle projected onto the unit direction (ux, uy).
inline double rect_half_extent(const Rect& r, double ux, double uy) {
    const double c = std::cos(r.yaw);
    const double s = std::sin(r.yaw);
    return 0.5 * (r.length * std::abs(ux * c + uy * s) + r.width * std::abs(-ux * s + uy * c));
}

} // namespace sf
