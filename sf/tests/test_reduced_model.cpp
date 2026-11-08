// Steady-state closures: saturation branch, closure/inverse consistency,
// rear friction-circle table, gamma-curve inverse, and the actuator delay
// line of step().

#include "sf_model.h"
#include "sf_tuning.h"
#include "vehicle/params.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

using namespace sf;

namespace {

// Top speed of the lab vehicle.
constexpr double kVMaxMps = 4.0;

int failures = 0;

void check(bool condition, const char *message) {
  std::printf("  [%s] %s\n", condition ? "PASS" : "FAIL", message);
  if (!condition) {
    ++failures;
  }
}

bool near(double value, double expected, double tolerance) {
  return std::abs(value - expected) <= tolerance;
}

} // namespace

int main() {
  std::printf("== reduced model ==\n");

  const double sat_25 = delta_sat_rad(2.5);
  check(near(sat_25, 0.2862, 2e-3), "delta_sat(v=2.5) ~ 0.2862");
  const SteadyStateCornering sat = steady_state_cornering(sat_25 + 0.01, 2.5);
  check(sat.saturated && near(sat.a_y, kLatAccelHiMps2, 1e-12),
        "saturated a_y = a_lat_hi");
  check(near(sat.r, sat.a_y / 2.5, 1e-9), "saturated r = a_y / v");

  bool inverse_ok = true;
  for (double v = 1.0; v <= 4.0; v += 0.5) {
    for (double a_y = -4.5; a_y <= 4.5; a_y += 0.5) {
      const double delta = delta_for_ay(a_y, v);
      if (std::abs(delta) < delta_sat_rad(v)) {
        inverse_ok =
            inverse_ok && near(steady_state_cornering(delta, v).a_y, a_y, 1e-6);
      }
    }
  }
  check(inverse_ok,
        "steady_state_cornering(delta_for_ay(a_y)) recovers a_y to 1e-6");
  check(near(steady_state_cornering(0.0, 3.0).a_y, 0.0, 1e-12),
        "straight wheel gives zero a_y");
  check(steady_state_cornering(0.1, 3.0).a_y >
            steady_state_cornering(0.05, 3.0).a_y,
        "a_y monotone in delta");

  check(delta_max_rad(1.0, SafetyParameters::lat_budget_mps2) <=
            kSteerLockTargetRad + 1e-12,
        "box <= lock");
  check(delta_max_rad(4.0, SafetyParameters::lat_budget_mps2) <
            delta_max_rad(2.0, SafetyParameters::lat_budget_mps2),
        "box shrinks with speed");

  const double coast = rear_lateral_accel_limit(0.0);
  check(near(coast, kLatAccelHiMps2, 1e-9), "rear_lateral_accel_limit(0) = a_lat_hi");
  check(rear_lateral_accel_limit(-1.0) < coast && rear_lateral_accel_limit(-2.0) < rear_lateral_accel_limit(-1.0),
        "kamm budget monotone");
  check(rear_lateral_accel_limit(-kMaxAccelMps2) / coast > 0.40 &&
            rear_lateral_accel_limit(-kMaxAccelMps2) / coast < 0.60,
        "full brake leaves ~half the lateral budget");

  bool gamma_ok = true;
  for (double cmd = -0.4; cmd <= 0.4; cmd += 0.05) {
    const double target =
        axle_target_from_command(cmd, kSteerDeltaThRad, kSteerGamma);
    gamma_ok =
        gamma_ok &&
        near(command_from_axle_target(target, kSteerDeltaThRad, kSteerGamma),
             cmd, 1e-8);
  }
  check(gamma_ok, "gamma inverse recovers the command to 1e-8");
  bool nominal_throws = false;
  try {
    assert_gamma_monotone(kSteerDeltaThRad, kSteerGamma);
  } catch (const std::logic_error &) {
    nominal_throws = true;
  }
  check(!nominal_throws, "nominal gamma curve is accepted as monotone");
  // A large gamma makes the command curve fall inside the command range.
  bool nonmonotone_throws = false;
  try {
    assert_gamma_monotone(kSteerDeltaThRad, 1000.0);
  } catch (const std::logic_error &) {
    nonmonotone_throws = true;
  }
  check(nonmonotone_throws, "non-monotone gamma curve is rejected");

  check(steer_switch_s() > 0.0 && steer_switch_s() < 2.0 * kControlPeriodS,
        "steer switch within two periods");
  check(speed_switch_s() > 0.0 && speed_switch_s() < steer_switch_s(),
        "speed switches before steer");
  State z{};
  z.v = 3.0;
  const State held = step(z, Input{0.0, 3.0}, Input{0.0, 3.0}, kControlPeriodS, kVMaxMps);
  check(near(held.delta, 0.0, 1e-12) && near(held.y, 0.0, 1e-12),
        "straight hold stays straight");
  // Publish 0.2 at t = 0: the steering follows it only from the switch time,
  // so after two periods it has moved less than under a two-period hold.
  const State turned_1 =
      step(z, Input{0.0, 3.0}, Input{0.0, 3.0}, Input{0.2, 3.0}, kControlPeriodS, kVMaxMps);
  const State turned_2 =
      step(turned_1, Input{0.0, 3.0}, Input{0.2, 3.0}, Input{0.2, 3.0}, kControlPeriodS, kVMaxMps);
  const State full_1 =
      step(z, Input{0.2, 3.0}, Input{0.2, 3.0}, Input{0.2, 3.0}, kControlPeriodS, kVMaxMps);
  const State full_2 =
      step(full_1, Input{0.2, 3.0}, Input{0.2, 3.0}, Input{0.2, 3.0}, kControlPeriodS, kVMaxMps);
  const bool inside = steer_switch_s() < kControlPeriodS;
  check(inside ? turned_1.delta > 0.0 : near(turned_1.delta, 0.0, 1e-12),
        "steering moves within the first period only if the switch is inside it");
  check(turned_2.delta > 0.0 && turned_2.delta < full_2.delta,
        "late switch delivers less steering than a full hold");
  // The two-command form is the three-command form with an unchanged history.
  const State split = step(z, Input{0.0, 3.0}, Input{0.2, 3.0}, kControlPeriodS, kVMaxMps);
  check(near(split.delta, turned_1.delta, 1e-12), "two-command step matches three-command step with equal history");
  const State braked =
      step(z, Input{0.0, 3.0}, Input{0.0, 2.0}, kControlPeriodS, kVMaxMps);
  check(braked.v < 3.0 && braked.v > 2.0,
        "speed setpoint below v decelerates within the period");
  const State rate = step(State{0, 0, 0, 0.0, 1.0}, Input{0.375, 1.0},
                          Input{0.375, 1.0}, kControlPeriodS, kVMaxMps);
  check(rate.delta <= 3.2 * kControlPeriodS + 1e-9,
        "steering rate limit honoured");

  std::printf("%s (%d failing checks)\n", failures ? "FAILURES" : "OK",
              failures);
  return failures ? 1 : 0;
}
