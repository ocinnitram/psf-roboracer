// Collision-cone barrier and terminal-set conditions.

#include "sf_barrier.h"
#include "vehicle/params.h"

#include <algorithm>
#include <cmath>

#include "sf_math.h"

namespace sf {

double disk_velocity_heading(const State &z, const SteadyStateCornering &c,
                             double l_i_m) {
  const double v = std::max(z.v, kKinematicFloorMps);
  return z.psi +
         std::atan2(v * std::sin(c.beta) + l_i_m * c.r, v * std::cos(c.beta));
}

double cone_h(double qx, double qy, double phi, double cx, double cy,
              double r_keep_m) {
  const double px = cx - qx;
  const double py = cy - qy;
  const double d = std::hypot(px, py);
  const double chi = wrap_pi(phi - std::atan2(py, px));
  const double half_angle = (d <= r_keep_m) ? kHalfPi : std::asin(r_keep_m / d);
  return std::abs(chi) - half_angle;
}

double escape_distance(double v_mps, double lateral_m, double a_lat_mps2) {
  return v_mps * (std::sqrt(2.0 * lateral_m / a_lat_mps2) + kSteerResponseS);
}

double lateral_stop_heading_limit(double a_perp_mps2, double room_m,
                                  double w_mps) {
  if (a_perp_mps2 <= 0.0 || room_m <= 0.0) {
    return 0.0;
  }
  if (w_mps <= 0.0) {
    return kHalfPi;
  }
  return std::asin(
      std::min(1.0, std::sqrt(2.0 * a_perp_mps2 * room_m) / w_mps));
}

double curvature_speed_limit(double kappa, double a_lat_mps2,
                             double v_max_mps) {
  if (std::abs(kappa) < 1e-9) {
    return v_max_mps;
  }
  return std::min(v_max_mps, std::sqrt(a_lat_mps2 / std::abs(kappa)));
}

double brake_room(double a_y_mps2, double a_lat_mps2) {
  const double a_y = std::min(std::abs(a_y_mps2), a_lat_mps2);
  return std::sqrt(std::max(0.0, a_lat_mps2 * a_lat_mps2 - a_y * a_y));
}

} // namespace sf
