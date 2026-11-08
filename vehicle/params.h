#pragma once

#include <cmath>

#include "sf_math.h"

// Shared measured and derived vehicle constants.

namespace sf {

// ---- Body and tyres ----
constexpr double kMassKg = 3.74;
constexpr double kLengthFrontM = 0.15875;
constexpr double kLengthRearM = 0.17145;
constexpr double kWheelbaseM = kLengthFrontM + kLengthRearM;
constexpr double kCogHeightM = 0.074;
// Nominal tyre parameters.
constexpr double kTyreS = 1.20;
constexpr double kTyreMu = 0.42;
constexpr double kAlphaPeakRad = 0.16;
constexpr double kCorneringStiffnessFront = 4.718;
constexpr double kCorneringStiffnessRear = 5.4562;
constexpr double kGravityMps2 = 9.81;
constexpr double kMaxAccelMps2 = 3.48;
constexpr double kMaxSpeedMps = 4.0;
constexpr double kSteerLockCmdRad = 0.4189;
constexpr double kSteerLockTargetRad = 0.3753;
constexpr double kTauSpeedPlantS = 0.118; // plant: a_des = (v_des - v) / tau_v
constexpr double kEgoLengthM = 0.50;
constexpr double kEgoWidthM = 0.29;

// Steering transmission: command to axle target.
constexpr double kSteerGamma = 0.622;
constexpr double kSteerDeltaThRad = 0.033;

// ---- Timing and actuator fits ----
constexpr double kControlPeriodS = 0.067;
constexpr double kOdomPeriodS = 0.020; // odometry publish period
// Fixed sense-to-act delay: the node publishes at scan time plus this, after
// policy, detector, tracker and filter have run, so the actuators see a
// constant delay. Set at the p99 of the measured scan-to-drive latency with
// the filter engaged.
constexpr double kComputeLatencyS = 0.020;
// Spread of that delay; zero because the publish time is scheduled, not
// whenever compute finishes. The sim draws the hold from it under sim2real.
constexpr double kComputeLatencySdS = 0.0;
// Model transport, steering-servo, and acceleration response.
constexpr double kSteerTransportS = 0.060;
constexpr double kSpeedTransportS = 0.020;
constexpr double kTauDeltaS = 0.075;
constexpr double kTauDeltaDotS = 0.027;
constexpr double kTauAccelS = 0.061;
constexpr double kSteerRateMaxRadps = 3.2;
// Steering response used by the backup timing and the escape distance: the
// two identified servo lags, tau_delta + tau_delta_dot. The transport delay is
// not included; the rollout and the backup's preview carry it explicitly.
constexpr double kSteerResponseS = kTauDeltaS + kTauDeltaDotS;
// Dynamic-model measurements and low-speed transition.
constexpr double kInertiaKgm2 = 0.04712;
constexpr double kSigmaFyM = 0.03;
constexpr double kKinematicFloorMps = 0.5;
constexpr double kTauBetaKinS = 0.15;
// Blend speed of the kinematic-dynamic single-track plant.
constexpr double kVSwitchMps = 7.319;

// Peak lateral acceleration of the nominal tyre.
constexpr double kLatAccelHiMps2 = kTyreS * kTyreMu * kGravityMps2;

// ---- LiDAR geometry measured forward from the CoG ----
constexpr double kLidarDistM = 0.10355;
// SICK LiDAR scan geometry published on /scan.
constexpr int kLidarNumBeams = 811;
constexpr double kLidarAngleIncRad = 0.0058172;
constexpr double kLidarFovRad = (kLidarNumBeams - 1) * kLidarAngleIncRad;
constexpr double kLidarAngleMinRad = -0.5 * kLidarFovRad;
constexpr double kLidarAngleMaxRad = 0.5 * kLidarFovRad;
// Range the filter and the sim trust; the sensor reports further.
constexpr double kLidarMaxRangeM = 10.0;

// Derived tyre shape.
inline double pacejka_shape_C() {
  return 2.0 * (kPi - std::asin(1.0 / kTyreS)) / kPi;
}
inline double pacejka_B_rear() {
  return std::tan(kPi / (2.0 * pacejka_shape_C())) / kAlphaPeakRad;
}
inline double pacejka_B_front() {
  return pacejka_B_rear() * kCorneringStiffnessFront / kCorneringStiffnessRear;
}
inline double understeer_c1() {
  return 1.0 / pacejka_B_front() - 1.0 / pacejka_B_rear();
}

} // namespace sf
