#pragma once

#include "obstacle_detector.h"

#include <vector>

namespace sf {

// 1-D scalar Kalman filter: F=1 (static model), H=1 (direct observation).
struct ScalarKF {
  double x; // state estimate
  double P; // estimate covariance
  double Q; // process noise per frame
  double R; // measurement noise

  ScalarKF(double x0, double P0, double Q_, double R_)
      : x(x0), P(P0), Q(Q_), R(R_) {}

  void predict() { P += Q; }

  void update(double z) {
    const double K = P / (P + R);
    x = x + K * (z - x);
    P = (1.0 - K) * P;
  }

  void reset(double z) {
    x = z;
    P = R;
  }
};

// Constant-velocity Kalman filter for one position axis measured directly.
struct ConstVelKF {
  double pos = 0.0;     // position estimate                     [m]
  double vel = 0.0;     // velocity estimate                     [m/s]
  double P_pos = 0.0;   // position covariance                    [m^2]
  double P_cross = 0.0; // cross covariance (pos, vel)           [m^2/s]
  double P_vel = 0.0;   // velocity covariance                    [m^2/s^2]
  double Q_vel = 0.0;   // velocity process noise per second      [m^2/s^3]

  ConstVelKF(double pos0, double P0_pos, double P0_vel, double Q_vel_)
      : pos(pos0), P_pos(P0_pos), P_vel(P0_vel), Q_vel(Q_vel_) {}

  void predict(double dt_s) {
    pos += vel * dt_s;
    P_pos += 2.0 * P_cross * dt_s + P_vel * dt_s * dt_s;
    P_cross += P_vel * dt_s;
    P_vel += Q_vel * dt_s;
  }

  void update(double z, double R) {
    const double S = P_pos + R;
    const double K_pos = P_pos / S;
    const double K_vel = P_cross / S;
    const double innov = z - pos;
    pos += K_pos * innov;
    vel += K_vel * innov;
    P_vel -= K_vel * P_cross;
    P_cross -= K_pos * P_cross;
    P_pos -= K_pos * P_pos;
  }
};

// Filtered position, velocity, extent, and yaw for one obstacle.
struct TrackedObstacle {
  int id;
  int age = 0;
  int missing_count = 0;

  // Track measured box extents independently; mirror the known extent until both are seen.
  static constexpr double kQDim = 0.0;
  static constexpr double kQYaw = 0.0;
  static constexpr double kRPos = 0.2; // validated on the 500-layout sweep
  static constexpr double kRDim = 0.0036; // face-extent std 0.06 m
  static constexpr double kRYaw = 0.2;
  static constexpr double kP0Pos = 1.00;
  static constexpr double kP0Yaw = 1.00;
  static constexpr double kDimGateChi2 = 6.63; // 99%, one degree of freedom
  ScalarKF kf_u, kf_v, kf_yaw;
  bool u_measured = false;
  bool v_measured = false;

  // Measured fit-centre noise in the line-of-sight frame
  // (evaluation/tools/measure_tracker_noise.py). The
  // range-independent across term is held at the along sigma, above its fit.
  static constexpr double kSigmaAlongM = 0.042;
  static constexpr double kSigmaAcrossM = 0.042;
  static constexpr double kSigmaHeadingRad = 0.012;
  static constexpr double kBiasAlongM = 0.030;

  // Models about 0.03 m/s of speed drift over one second.
  static constexpr double kQVel = 0.001; // velocity process noise   [m^2/s^3]
  static constexpr double kP0Vel = 1.0; // initial velocity covariance [m^2/s^2]
  ConstVelKF cv_x, cv_y;

  // Latch velocity after the 2-DOF significance test passes.
  static constexpr double kVelocityChi2 = 23.0;
  // Suppress false motion from rotating fit bias while passing nearby.
  static constexpr double kVelocityRangeMinM = 1.5;
  bool moving = false;

  // Reopen velocity covariance when recent innovations reject the static model.
  static constexpr int kOnsetWindow = 5;
  static constexpr double kOnsetChi2 = 27.6;
  double innov_al[kOnsetWindow] = {};
  double innov_ac[kOnsetWindow] = {};
  int innov_count = 0;

  TrackedObstacle(int id, const RawDetection &det, const Pose2D &pose);

  void predict(double dt_s);

  // Refresh this track from one associated detection.
  void update(const RawDetection &det, const Pose2D &pose);
  void update_yaw(double yaw_fit);

  double x() const { return cv_x.pos; }
  double y() const { return cv_y.pos; }
  double vx() const { return moving ? cv_x.vel : 0.0; }
  double vy() const { return moving ? cv_y.vel : 0.0; }
  double u() const { return u_measured ? kf_u.x : kf_v.x; }
  double v() const { return v_measured ? kf_v.x : kf_u.x; }
  double yaw() const { return kf_yaw.x; }
};

// Associate detections over time and emit filtered obstacle rectangles.
class ObstacleTracker {
public:
  ObstacleTracker();

  std::vector<Rect> update(const RawDetections &raw_detections);
  void reset();

private:
  void predict_tracks();
  std::vector<int> associate_detections(const std::vector<RawDetection> &items);
  void apply_matches(const std::vector<RawDetection> &items,
                     const std::vector<int> &assignment, const Pose2D &pose);
  void prune_missing();
  std::vector<Rect> emit_confirmed() const;

  std::vector<TrackedObstacle> m_obstacles;
  int m_next_id = 0;

  static constexpr double kChi2Gate = 9.21; // 99% confidence, 2-DOF chi-squared
  static constexpr int kMaxMissingFrames =
      5; // prune after this many missed frames
  static constexpr int kMinConfirmAge = 2; // don't report until age >= this
};

} // namespace sf
