// Collision-cone barrier and terminal-set conditions.
//   1. The bearing form of the cone and the speed-scaled form define
//      the same set: zero sign disagreements over 4000 random states.
//   2. Nagumo condition: holding the wheel straight on a tangent keeps
//      h on the boundary over 6 periods of step().
//   3. Escape distance is linear in v and matches the closed form.
//   4. Lateral stopping: heading limit shrinks with speed and available room.
//   5. Speed limit: v_max on a straight, sqrt(a / |kappa|) otherwise.

#include "sf_barrier.h"
#include "sf_tuning.h"
#include "vehicle/params.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "sf_math.h"

using namespace sf;

namespace {

// Top speed of the lab vehicle.
constexpr double kVMaxMps = 4.0;

double speed_scaled_h(double qx, double qy, double phi, double speed, double cx,
                      double cy, double r_keep_m) {
  const double px = cx - qx;
  const double py = cy - qy;
  const double d = std::hypot(px, py);
  const double chi = wrap_pi(phi - std::atan2(py, px));
  const double radial = std::sqrt(std::max(0.0, d * d - r_keep_m * r_keep_m));
  return speed * (radial - d * std::cos(chi));
}

int failures = 0;

void check(bool condition, const char *message) {
  std::printf("  [%s] %s\n", condition ? "PASS" : "FAIL", message);
  if (!condition) {
    ++failures;
  }
}

double uniform(double lo, double hi) {
  return lo + (hi - lo) * static_cast<double>(std::rand()) / RAND_MAX;
}

} // namespace

int main() {
  std::printf("== cone barrier ==\n");
  const double r_keep = 0.45;

  std::srand(7);
  int disagreements = 0;
  for (int i = 0; i < 4000; ++i) {
    const double phi = uniform(-kPi, kPi);
    const double d = uniform(r_keep + 0.05, 8.0);
    const double bearing = uniform(-kPi, kPi);
    const double cx = d * std::cos(bearing);
    const double cy = d * std::sin(bearing);
    const double h = cone_h(0.0, 0.0, phi, cx, cy, r_keep);
    const double hp =
        speed_scaled_h(0.0, 0.0, phi, uniform(1.0, 4.0), cx, cy, r_keep);
    if (std::abs(h) > 1e-9 && (h > 0.0) != (hp > 0.0)) {
      ++disagreements;
    }
  }
  check(disagreements == 0, "bearing and speed-scaled cone forms agree in sign "
                            "on 4000 random states");

  State z{};
  z.v = 3.0;
  const double d0 = 2.0;
  const double chi0 = std::asin(r_keep / d0);
  const double cx = d0 * std::cos(-chi0);
  const double cy = d0 * std::sin(-chi0);
  double worst = 0.0;
  for (int k = 0; k < 6; ++k) {
    const SteadyStateCornering c = steady_state_cornering(z.delta, z.v);
    const double phi = disk_velocity_heading(z, c, 0.0);
    worst = std::max(worst, std::abs(cone_h(z.x, z.y, phi, cx, cy, r_keep)));
    z = step(z, Input{0.0, 3.0}, Input{0.0, 3.0}, kControlPeriodS, kVMaxMps);
  }
  check(worst < 1e-9, "tangent pass holds h = 0 over 6 periods");

  // Sideslip rotates the velocity off the body axis; the disk heading must
  // carry it (psi + beta at l = 0), plus the yaw-rate term at an offset.
  {
    State s{};
    s.psi = 0.4;
    s.v = 3.0;
    SteadyStateCornering c{};
    c.beta = 0.2;
    c.r = 1.5;
    const double at_cog = disk_velocity_heading(s, c, 0.0);
    const double at_front = disk_velocity_heading(s, c, 0.3);
    const double expect_front =
        s.psi + std::atan2(s.v * std::sin(c.beta) + 0.3 * c.r,
                           s.v * std::cos(c.beta));
    check(std::abs(at_cog - (s.psi + c.beta)) < 1e-12,
          "disk heading at the CoG is psi + beta");
    check(std::abs(at_front - expect_front) < 1e-12 && at_front > at_cog,
          "yaw rate adds to the heading of a forward disk");
  }

  const double t_esc =
      std::sqrt(2.0 * r_keep / SafetyParameters::lat_budget_mps2) + kSteerResponseS;
  check(std::abs(escape_distance(2.0, r_keep, SafetyParameters::lat_budget_mps2) -
                 2.0 * t_esc) < 1e-12 &&
            std::abs(escape_distance(4.0, r_keep, SafetyParameters::lat_budget_mps2) -
                     4.0 * t_esc) < 1e-12,
        "escape distance is v times the escape time");

  check(std::abs(lateral_stop_heading_limit(4.0, 0.5, 1.0) - kHalfPi) <
            1e-12,
        "heading is unrestricted at low speed");
  check(lateral_stop_heading_limit(4.0, 0.5, 4.0) <
            lateral_stop_heading_limit(4.0, 0.5, 3.0),
        "heading limit shrinks with speed");
  check(lateral_stop_heading_limit(0.0, 0.5, 3.0) == 0.0 &&
            lateral_stop_heading_limit(4.0, 0.0, 3.0) == 0.0,
        "heading limit is zero without lateral budget or room");

  check(curvature_speed_limit(0.0, 4.0, kVMaxMps) == kVMaxMps,
        "corridor speed is v_max on a straight");
  check(std::abs(curvature_speed_limit(0.3, 4.121, kVMaxMps) - 3.706) < 1e-2,
        "corridor speed is 3.71 m/s at |kappa| 0.30");

  check(brake_room(0.0, 4.0) == 4.0 && brake_room(-3.0, 5.0) == 4.0,
        "braking room is the friction circle's longitudinal leg");
  check(brake_room(4.0, 4.0) == 0.0 && brake_room(6.0, 4.0) == 0.0,
        "no braking at or beyond the lateral budget");

  std::printf("%s (%d failing checks)\n", failures ? "FAILURES" : "OK",
              failures);
  return failures ? 1 : 0;
}
