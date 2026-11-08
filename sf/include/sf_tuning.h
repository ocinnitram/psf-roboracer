#pragma once

#include "vehicle/params.h"

#include <algorithm>

// Filter operating point, collision margins, horizon, and solver numerics.
// Vehicle quantities are in vehicle/params.h.

namespace sf {

// Measurements, derived limits, and validated backup operating points.
struct SafetyParameters {
  // Limit backup control to the measured usable fraction of tyre grip.
  static constexpr double grip_eta = 0.95;
  static constexpr double lat_budget_mps2 = grip_eta * kLatAccelHiMps2;

  // Backup operating point validated on the 500-layout evaluation set.
  static constexpr double brake_pass_mps2 = 2.0;
  static constexpr double cruise_mps = 1.5;
  static constexpr double max_crossing_sin = 0.45;

  // Derive backup loop timing from the steering response.
  static constexpr double lateral_tau_s = 2.0 * kSteerResponseS;
  static constexpr double time_s = 3.0 * kSteerResponseS;
  static constexpr double look_s = kSteerResponseS + lateral_tau_s;
  static constexpr double settle_s =
      3.0 * time_s + lateral_tau_s + kSteerResponseS;

  // Cover measured tracker, odometry, and map-quantisation errors.
  static constexpr double obstacle_margin_m = 0.10;
  static constexpr double wall_margin_m = 0.08;
  // Bound tracker velocity error measured in moving-box trials.
  static constexpr double obstacle_vel_sigma_mps = 0.06;
  // Treat reported speeds below three sigma as static.
  static constexpr double moving_speed_floor_mps = 3.0 * obstacle_vel_sigma_mps;

  // Scale the certificate margin with measured pose and timing errors.
  static constexpr double certify_margin_m = 0.03;
  static constexpr double certify_margin_s = 0.015;
  // Against walls the speed term is doubled: heading noise carried over the
  // horizon lets a car on the certificate's edge reach the outer wall of a
  // hairpin (validated on the 500-layout set over six noise seeds).
  static constexpr double certify_wall_margin_s = 0.03;

  // Scale steering and speed deviations in the QP objective.
  static constexpr double qp_steer_scale_rad = 0.20;
  static constexpr double qp_speed_scale_mps = 0.25;

  // Use enough body disks to keep approximation error below odometry noise.
  static constexpr int cover_disks = 4;
};

// Default horizon, set by the control-tick compute budget. A
// top speed that needs longer to stop extends the horizon past this floor so
// the certificate can still reach a stop.
constexpr int kHorizonFloorStages = 24;

// Filter limits that scale with the top speed it is configured for, derived
// once when the filter is constructed.
struct SpeedLimits {
  explicit SpeedLimits(double v_max_mps)
      : v_max(v_max_mps),
        // Inset the backup reference by the top-speed certificate margin.
        reference_inset_m(SafetyParameters::certify_margin_m +
                          SafetyParameters::certify_margin_s * v_max_mps),
        // Simulate long enough to stop from the top speed.
        min_stages(
            static_cast<int>(v_max_mps / kMaxAccelMps2 / kControlPeriodS) + 1),
        max_stages(std::max(kHorizonFloorStages, min_stages)),
        // Liveness: a changed command may lose at most one speed unit held
        // over the minimum horizon of progress against the backup command.
        progress_tol_m(SafetyParameters::qp_speed_scale_mps * min_stages *
                       kControlPeriodS) {}

  // Declaration order is the initialisation order the constructor relies on.
  double v_max;
  double reference_inset_m;
  int min_stages;
  int max_stages;
  double progress_tol_m;
};

inline double certify_margin(double v_mps) {
  return SafetyParameters::certify_margin_m +
         SafetyParameters::certify_margin_s * v_mps;
}

inline double certify_wall_margin(double v_mps) {
  return SafetyParameters::certify_margin_m +
         SafetyParameters::certify_wall_margin_s * v_mps;
}

// ---- Numerics ----
// Numerical steps and iteration limits for filtering and integration.
constexpr double kGradStepSteerRad = 0.02;
constexpr double kGradStepSpeedMps = 0.10;
constexpr int kSqpIters = 2;
constexpr int kSegmentHalvings = 5;
constexpr double kQpInterior = 0.01;
constexpr int kNewtonAyIters = 12;
constexpr double kNewtonAyAbsTol = 1e-12;
constexpr int kGammaInverseIters = 60;
constexpr int kSubstepsPerPeriod = 2;

} // namespace sf
