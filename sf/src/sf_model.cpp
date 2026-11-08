// Dynamic single-track prediction with actuators and steady-state cornering.

#include "sf_model.h"
#include "core_math.h"
#include "sf_math.h"
#include "sf_tuning.h"
#include "vehicle/params.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sf {

namespace {

// Differentiate tyre response by lateral acceleration.
double tyre_response_derivative(double a_y_mps2) {
  const double ratio = a_y_mps2 / kLatAccelHiMps2;
  const double t = tyre_response_from_lateral_accel(a_y_mps2);
  return (1.0 + t * t) /
         (pacejka_shape_C() * kLatAccelHiMps2 * std::sqrt(1.0 - ratio * ratio));
}

// Keep lateral acceleration inside the tyre limit and away from singularity.
double clamp_ay(double a_y_mps2) {
  const double limit = kLatAccelHiMps2 - 1e-9;
  return std::max(-limit, std::min(limit, a_y_mps2));
}

// Restrict a scalar to an inclusive interval.
double clamp(double x, double lo, double hi) {
  return std::max(lo, std::min(hi, x));
}

struct Loads {
  double f_zf;
  double f_zr;
};

// Compute front and rear vertical loads under longitudinal acceleration.
Loads loads_at(double accel) {
  return {kMassKg * (kGravityMps2 * kLengthRearM - accel * kCogHeightM) /
              kWheelbaseM,
          kMassKg * (kGravityMps2 * kLengthFrontM + accel * kCogHeightM) /
              kWheelbaseM};
}

// Compute nominal steady-state Pacejka force at each axle; `slip` is
// rotation(z.beta).
void steady_forces(const State &z, const Rotation &slip, double accel,
                   double *fy_f, double *fy_r) {
  const double vx = z.v * slip.c;
  if (vx < kKinematicFloorMps) {
    *fy_f = *fy_r = 0.0;
    return;
  }
  const double vy = z.v * slip.s;
  const double alpha_f = z.delta - std::atan2(vy + kLengthFrontM * z.r, vx);
  const double alpha_r = -std::atan2(vy - kLengthRearM * z.r, vx);
  const Loads w = loads_at(accel);
  const double grip = kTyreS * kTyreMu;
  *fy_f = grip * w.f_zf *
          std::sin(pacejka_shape_C() * cr_atan(pacejka_B_front() * alpha_f));
  *fy_r = grip * w.f_zr *
          std::sin(pacejka_shape_C() * cr_atan(pacejka_B_rear() * alpha_r));
}

// Steering-rate and acceleration derivatives: rate-limited second-order
// servo toward the axle target, first-order lag toward the speed demand.
void actuator_rates(const State &z, double delta_t, double v_des,
                    double *delta_ddot, double *jerk) {
  const double delta_dot_des = (delta_t - z.delta) / kTauDeltaS;
  *delta_ddot = (delta_dot_des - z.delta_dot) / kTauDeltaDotS;
  const double a_des =
      clamp((v_des - z.v) / kTauSpeedPlantS, -kMaxAccelMps2, kMaxAccelMps2);
  *jerk = (a_des - z.accel) / kTauAccelS;
}

// Compute the state derivative under the commanded axle target and speed;
// the actuator states are integrated together with the body. `slip` is
// rotation(z.beta).
State dynamics(const State &z, const Rotation &slip, double delta_t,
               double v_des) {
  State dz;
  const double sv = z.delta_dot;
  const double accel = z.accel;
  actuator_rates(z, delta_t, v_des, &dz.delta_dot, &dz.accel);
  const double vx = z.v * slip.c;
  if (vx < kKinematicFloorMps) {
    dz.x = z.v * std::cos(z.psi);
    dz.y = z.v * std::sin(z.psi);
    dz.psi = z.v / kWheelbaseM * std::tan(z.delta);
    dz.delta = sv;
    dz.v = accel;
    const double c = std::cos(z.delta);
    dz.r = accel / kWheelbaseM * std::tan(z.delta) +
           z.v / (kWheelbaseM * c * c) * sv;
    dz.beta = -z.beta / kTauBetaKinS;
    return dz;
  }
  const double vy = z.v * slip.s;
  const Loads w = loads_at(accel);
  // Split longitudinal demand by load and cap each axle by its friction circle.
  const double fx_total = kMassKg * accel;
  const double fx_f0 = fx_total * w.f_zf / (w.f_zf + w.f_zr);
  const double fx_r0 = fx_total - fx_f0;
  const double cap_f = kTyreS * kTyreMu * w.f_zf;
  const double cap_r = kTyreS * kTyreMu * w.f_zr;
  const double n_f = std::hypot(fx_f0, z.fy_f);
  const double n_r = std::hypot(fx_r0, z.fy_r);
  const double s_f = (n_f > cap_f) ? cap_f / n_f : 1.0;
  const double s_r = (n_r > cap_r) ? cap_r / n_r : 1.0;
  const double fx_f = fx_f0 * s_f;
  const double fy_f = z.fy_f * s_f;
  const double fx_r = fx_r0 * s_r;
  const double fy_r = z.fy_r * s_r;
  const double cd = std::cos(z.delta);
  const double sd = std::sin(z.delta);
  const double vx_dot = (fx_f * cd - fy_f * sd + fx_r) / kMassKg + vy * z.r;
  const double vy_dot = (fx_f * sd + fy_f * cd + fy_r) / kMassKg - vx * z.r;
  dz.r = (kLengthFrontM * (fy_f * cd + fx_f * sd) - kLengthRearM * fy_r) /
         kInertiaKgm2;
  dz.v = (vx * vx_dot + vy * vy_dot) / z.v;
  dz.beta = (vx * vy_dot - vy * vx_dot) / (z.v * z.v);
  dz.x = z.v * std::cos(z.beta + z.psi);
  dz.y = z.v * std::sin(z.beta + z.psi);
  dz.psi = z.r;
  dz.delta = sv;
  return dz;
}

// Add a scaled mechanical-state derivative to a state.
State add_scaled_derivative(const State &z, double a, const State &k) {
  State out = z;
  out.x += a * k.x;
  out.y += a * k.y;
  out.psi += a * k.psi;
  out.delta += a * k.delta;
  out.v += a * k.v;
  out.r += a * k.r;
  out.beta += a * k.beta;
  out.delta_dot += a * k.delta_dot;
  out.accel += a * k.accel;
  return out;
}

// Advance actuators, tyre relaxation, and body dynamics by one RK4 substep.
// The servo constant (27 ms) is shorter than the substep, so the actuator
// states are integrated inside the RK4 rather than held over it; their
// saturations are applied to the result, as the plant does every millisecond.
State substep(const State &z0, double delta_t, double v_des, double h_s,
              double v_max_mps) {
  State z = z0;
  // Relaxed lateral forces: exact first-order update toward steady state.
  double ss_f = 0.0;
  double ss_r = 0.0;
  const Rotation slip = rotation(z.beta);
  steady_forces(z, slip, z.accel, &ss_f, &ss_r);
  const double tau = kSigmaFyM / std::max(z.v * slip.c, kKinematicFloorMps);
  const double gain = 1.0 - std::exp(-h_s / tau);
  z.fy_f += gain * (ss_f - z.fy_f);
  z.fy_r += gain * (ss_r - z.fy_r);
  // RK4 on the mechanical and actuator states.
  const State k1 = dynamics(z, slip, delta_t, v_des);
  const State z2 = add_scaled_derivative(z, 0.5 * h_s, k1);
  const State k2 = dynamics(z2, rotation(z2.beta), delta_t, v_des);
  const State z3 = add_scaled_derivative(z, 0.5 * h_s, k2);
  const State k3 = dynamics(z3, rotation(z3.beta), delta_t, v_des);
  const State z4 = add_scaled_derivative(z, h_s, k3);
  const State k4 = dynamics(z4, rotation(z4.beta), delta_t, v_des);
  State out = z;
  out.x += h_s / 6.0 * (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x);
  out.y += h_s / 6.0 * (k1.y + 2.0 * k2.y + 2.0 * k3.y + k4.y);
  out.psi = wrap_pi(
      z.psi + h_s / 6.0 * (k1.psi + 2.0 * k2.psi + 2.0 * k3.psi + k4.psi));
  out.delta = clamp(
      z.delta +
          h_s / 6.0 * (k1.delta + 2.0 * k2.delta + 2.0 * k3.delta + k4.delta),
      -kSteerLockTargetRad, kSteerLockTargetRad);
  out.v =
      std::max(0.0, z.v + h_s / 6.0 * (k1.v + 2.0 * k2.v + 2.0 * k3.v + k4.v));
  out.r = z.r + h_s / 6.0 * (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r);
  out.beta =
      z.beta + h_s / 6.0 * (k1.beta + 2.0 * k2.beta + 2.0 * k3.beta + k4.beta);
  out.delta_dot = clamp(
      z.delta_dot + h_s / 6.0 *
                        (k1.delta_dot + 2.0 * k2.delta_dot +
                         2.0 * k3.delta_dot + k4.delta_dot),
      -kSteerRateMaxRadps, kSteerRateMaxRadps);
  if ((out.delta <= -kSteerLockTargetRad && out.delta_dot < 0.0) ||
      (out.delta >= kSteerLockTargetRad && out.delta_dot > 0.0)) {
    out.delta_dot = 0.0;
  }
  out.accel = clamp(
      z.accel + h_s / 6.0 *
                    (k1.accel + 2.0 * k2.accel + 2.0 * k3.accel + k4.accel),
      -kMaxAccelMps2, kMaxAccelMps2);
  if ((out.v <= 0.0 && out.accel < 0.0) ||
      (out.v >= v_max_mps && out.accel > 0.0)) {
    out.accel = 0.0;
  }
  return out;
}

// One segment of a control period under fixed commands.
State integrate_segment(const State &z, double delta_t, double v_des,
                        double h_s, double v_max_mps) {
  const double h_max = kControlPeriodS / kSubstepsPerPeriod;
  const int n = std::max(1, static_cast<int>(std::ceil(h_s / h_max - 1e-9)));
  State out = z;
  for (int i = 0; i < n; ++i) {
    out = substep(out, delta_t, v_des, h_s / n, v_max_mps);
  }
  return out;
}

} // namespace

// Wrap an angle to [-pi, pi]. The remainder is exact, so an angle already in
// range comes back unchanged and the result does not depend on the libm.
double wrap_pi(double angle_rad) {
  return std::remainder(angle_rad, 2.0 * kPi);
}

// Evaluate steady-state tyre response at a lateral acceleration.
double tyre_response_from_lateral_accel(double a_y_mps2) {
  return std::tan(std::asin(a_y_mps2 / kLatAccelHiMps2) / pacejka_shape_C());
}

// Return the steering angle where steady-state lateral acceleration saturates.
double delta_sat_rad(double v_mps) {
  const double v = std::max(v_mps, kKinematicFloorMps);
  return understeer_c1() *
             tyre_response_from_lateral_accel(clamp_ay(kLatAccelHiMps2)) +
         kWheelbaseM * kLatAccelHiMps2 / (v * v);
}

// Convert a steady-state lateral acceleration to an axle steering angle.
double delta_for_ay(double a_y_mps2, double v_mps) {
  const double a_y = clamp_ay(a_y_mps2);
  const double v = std::max(v_mps, kKinematicFloorMps);
  return understeer_c1() * tyre_response_from_lateral_accel(a_y) +
         kWheelbaseM * a_y / (v * v);
}

// Solve the steady-state cornering response for steering angle and speed.
SteadyStateCornering steady_state_cornering(double delta_rad, double v_mps) {
  SteadyStateCornering c;
  const double v = std::max(v_mps, kKinematicFloorMps);
  if (std::abs(delta_rad) >= delta_sat_rad(v)) {
    c.a_y = std::copysign(kLatAccelHiMps2, delta_rad);
    c.saturated = true;
  } else {
    // Newton on the steering equation, seeded at the kinematic root.
    double a_y = clamp_ay(delta_rad * v * v / kWheelbaseM);
    for (int iter = 0; iter < kNewtonAyIters; ++iter) {
      const double residual =
          understeer_c1() * tyre_response_from_lateral_accel(a_y) +
                              kWheelbaseM * a_y / (v * v) - delta_rad;
      const double slope =
          understeer_c1() * tyre_response_derivative(a_y) +
          kWheelbaseM / (v * v);
      a_y = clamp_ay(a_y - residual / slope);
      if (std::abs(residual) < kNewtonAyAbsTol) {
        break;
      }
    }
    c.a_y = a_y;
  }
  c.beta = kLengthRearM * c.a_y / (v * v) -
           tyre_response_from_lateral_accel(clamp_ay(c.a_y)) /
               pacejka_B_rear();
  c.r = c.a_y / v;
  return c;
}

// Limit steering to the angle allowed by grip and steering lock.
double delta_max_rad(double v_mps, double a_lat_mps2) {
  return std::min(kSteerLockTargetRad, delta_for_ay(a_lat_mps2, v_mps));
}

// Compute rear-axle lateral capacity remaining under longitudinal demand.
double rear_lateral_accel_limit(double a_x_mps2) {
  const double f_zr = kMassKg *
                      (kGravityMps2 * kLengthFrontM + a_x_mps2 * kCogHeightM) /
                      kWheelbaseM;
  const double f_xr = a_x_mps2 * f_zr / kGravityMps2;
  const double grip = (kLatAccelHiMps2 / kGravityMps2) * f_zr;
  const double radicand = std::max(0.0, grip * grip - f_xr * f_xr);
  return kWheelbaseM / (kMassKg * kLengthFrontM) * std::sqrt(radicand);
}

// Map a policy steering command through the steering transmission.
double axle_target_from_command(double delta_cmd_rad, double delta_th_rad,
                                double steer_gamma) {
  const double sq = delta_cmd_rad * delta_cmd_rad;
  if (sq == 0.0) {
    return 0.0;
  }
  return delta_cmd_rad * sq /
         ((sq + delta_th_rad * delta_th_rad) * (1.0 + steer_gamma * sq));
}

// Invert the steering transmission by bisection.
double command_from_axle_target(double delta_t_rad, double delta_th_rad,
                                double steer_gamma) {
  if (delta_t_rad == 0.0 ||
      (delta_th_rad == 0.0 && std::abs(steer_gamma) < 1e-12)) {
    return delta_t_rad;
  }
  const double bound = std::abs(delta_t_rad) + 1.0;
  double lo = -bound;
  double hi = bound;
  for (int iter = 0; iter < kGammaInverseIters; ++iter) {
    const double mid = 0.5 * (lo + hi);
    if (axle_target_from_command(mid, delta_th_rad, steer_gamma) <
        delta_t_rad) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return 0.5 * (lo + hi);
}

// Reject steering parameters whose command curve is not strictly increasing.
void assert_gamma_monotone(double delta_th_rad, double steer_gamma) {
  constexpr int kGridPoints = 4096;
  const double h = 2.0 * kSteerLockCmdRad / (kGridPoints - 1);
  double prev =
      axle_target_from_command(-kSteerLockCmdRad, delta_th_rad, steer_gamma);
  for (int i = 1; i < kGridPoints; ++i) {
    const double value = axle_target_from_command(-kSteerLockCmdRad + i * h,
                                                  delta_th_rad, steer_gamma);
    if (value <= prev) {
      throw std::logic_error(
          "gamma curve is not strictly monotone on the command range");
    }
    prev = value;
  }
}

// Return when a newly published steering command reaches the actuator.
double steer_switch_s() { return kComputeLatencyS + kSteerTransportS; }

// Return when a newly published speed command reaches the actuator.
double speed_switch_s() { return kComputeLatencyS + kSpeedTransportS; }

// Initialize unmeasured state from the measured state and active command.
void complete_state(State *z, const Input &active) {
  // Infer sideslip from measured yaw rate so existing rear slip enters the
  // rollout.
  const double v = std::max(z->v, kKinematicFloorMps);
  const double a_y = clamp_ay(z->r * v);
  const double beta =
      kLengthRearM * a_y / (v * v) -
      tyre_response_from_lateral_accel(a_y) / pacejka_B_rear();
  z->beta = beta * std::min(1.0, z->v / kKinematicFloorMps);
  z->delta_dot = 0.0;
  z->accel = clamp((active.v_des - z->v) / kTauSpeedPlantS, -kMaxAccelMps2,
                   kMaxAccelMps2);
  steady_forces(*z, rotation(z->beta), z->accel, &z->fy_f, &z->fy_r);
}

// Advance the model through one period with delayed command switching.
State step(const State &z, const Input &before, const Input &prev,
           const Input &next, double dt_s, double v_max_mps) {
  // A command published at the start of a period reaches an actuator after
  // its switch time, which may fall in the following period: within this
  // period the actuator follows `before` until (switch - dt), `prev` until
  // the switch, then `next`.
  const double t_s = steer_switch_s();
  const double t_v = speed_switch_s();
  double bounds[6] = {0.0, t_s - dt_s, t_v - dt_s, t_s, t_v, dt_s};
  for (double &b : bounds) {
    b = std::max(0.0, std::min(dt_s, b));
  }
  std::sort(bounds, bounds + 6);
  State out = z;
  for (int i = 0; i < 5; ++i) {
    const double a = bounds[i];
    const double b = bounds[i + 1];
    if (b <= a) {
      continue;
    }
    const double delta_t =
        (a < t_s - dt_s) ? before.delta_t : (a < t_s) ? prev.delta_t : next.delta_t;
    const double v_des =
        (a < t_v - dt_s) ? before.v_des : (a < t_v) ? prev.v_des : next.v_des;
    out = integrate_segment(out, delta_t, v_des, b - a, v_max_mps);
  }
  return out;
}

State step(const State &z, const Input &prev, const Input &next, double dt_s,
           double v_max_mps) {
  return step(z, prev, prev, next, dt_s, v_max_mps);
}

} // namespace sf
