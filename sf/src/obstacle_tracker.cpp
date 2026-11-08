// Associate detector boxes over time and emit filtered obstacle rectangles.

#include "obstacle_tracker.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "vehicle/params.h"
#include "sf_math.h"

namespace sf {

namespace {

constexpr double kSqrt1_2 = 0.70710678118654752440;
constexpr double kMinRangeM = 1e-9;

// Wrap an angle difference to [-pi/4, pi/4): a rectangle's axes repeat every
// quarter turn, so which fitted side is longer never flips the yaw.
double wrap_quarter_turn(double dyaw) {
  return dyaw - kHalfPi * std::floor(dyaw / kHalfPi + 0.5);
}

// Identify the box depth axis and orient it from the LiDAR toward the box.
struct DepthAxis {
  bool along_u; // true when the depth runs along the tracked yaw
  double sign;  // +1 when the box centre sits along +axis from the ego
  double ax;    // depth-axis unit vector, ego->box
  double ay;
};

// Select the box axis most aligned with the LiDAR line of sight.
DepthAxis select_depth_axis(double rx, double ry, double yaw_u) {
  const double cu = std::cos(yaw_u);
  const double su = std::sin(yaw_u);
  const double au = rx * cu + ry * su;
  const double av = -rx * su + ry * cu;
  const bool along_u = std::abs(au) >= std::abs(av);
  const double sign =
      along_u ? (au >= 0.0 ? 1.0 : -1.0) : (av >= 0.0 ? 1.0 : -1.0);
  return {along_u, sign, along_u ? cu : -su, along_u ? su : cu};
}

// Hold box-frame extents and whether each was directly measured.
struct ShapeMeasurement {
  DepthAxis axis;
  double u_ext, v_ext;
  bool u_measured, v_measured;
};

// Project a detection's fitted dimensions onto the tracked box axes.
ShapeMeasurement measure_shape(const RawDetection &det, const Pose2D &pose,
                               double yaw_tracked) {
  const DepthAxis axis = select_depth_axis(det.rect.cx - pose.x,
                                           det.rect.cy - pose.y, yaw_tracked);
  // Map the fitted dimensions (length along rect.yaw) onto the tracked axes.
  const bool length_along_u =
      std::abs(std::cos(det.rect.yaw - yaw_tracked)) >= kSqrt1_2;
  const double u_ext = length_along_u ? det.rect.length : det.rect.width;
  const double v_ext = length_along_u ? det.rect.width : det.rect.length;
  const bool both = !det.single_side_estimate;
  return {axis, u_ext, v_ext, both || !axis.along_u, both || axis.along_u};
}

// Initialize or innovation-gate one measured box extent.
void update_extent(ScalarKF *kf, bool *measured, double z) {
  if (!*measured) {
    kf->reset(z);
    *measured = true;
    return;
  }
  const double innov = z - kf->x;
  if (innov * innov < TrackedObstacle::kDimGateChi2 * (kf->P + kf->R)) {
    kf->update(z);
  }
}

struct Measurement2D {
  double x, y;
};

// Correct a fitted centre for tracked depth and line-of-sight bias.
Measurement2D position_measurement(const RawDetection &det, const Pose2D &pose,
                                   const ShapeMeasurement &m,
                                   double depth_tracked) {
  const double depth_det = m.axis.along_u ? m.u_ext : m.v_ext;
  const double corr = 0.5 * (depth_tracked - depth_det) * m.axis.sign;
  const double cx = det.rect.cx + corr * m.axis.ax;
  const double cy = det.rect.cy + corr * m.axis.ay;
  const double range = std::hypot(cx - pose.x, cy - pose.y);
  const double bias =
      range > kMinRangeM ? TrackedObstacle::kBiasAlongM / range : 0.0;
  return {cx + bias * (cx - pose.x), cy + bias * (cy - pose.y)};
}

// Reopen velocity uncertainty when innovations reject the static model.
void refresh_onset(TrackedObstacle &obs, double dx, double dy, double ux,
                   double uy, double sig_al, double sig_ac, double range) {
  const int slot = obs.innov_count % TrackedObstacle::kOnsetWindow;
  obs.innov_al[slot] = (dx * ux + dy * uy) / sig_al;
  obs.innov_ac[slot] = (-dx * uy + dy * ux) / sig_ac;
  obs.innov_count++;
  if (obs.innov_count < TrackedObstacle::kOnsetWindow ||
      range < TrackedObstacle::kVelocityRangeMinM) {
    return;
  }
  double mean_al = 0.0;
  double mean_ac = 0.0;
  for (int k = 0; k < TrackedObstacle::kOnsetWindow; ++k) {
    mean_al += obs.innov_al[k];
    mean_ac += obs.innov_ac[k];
  }
  mean_al /= TrackedObstacle::kOnsetWindow;
  mean_ac /= TrackedObstacle::kOnsetWindow;
  if (TrackedObstacle::kOnsetWindow * (mean_al * mean_al + mean_ac * mean_ac) >
      TrackedObstacle::kOnsetChi2) {
    obs.cv_x.P_vel = TrackedObstacle::kP0Vel;
    obs.cv_y.P_vel = TrackedObstacle::kP0Vel;
    obs.cv_x.P_cross = 0.0;
    obs.cv_y.P_cross = 0.0;
    obs.innov_count = 0;
  }
}

} // namespace

// Initialize a track's shape and position from its first detection. The
// detector's yaw prior (the track tangent) seeds the yaw filter and the fitted
// yaw is its first measurement; without a prior the fitted yaw seeds it.
TrackedObstacle::TrackedObstacle(int id, const RawDetection &det,
                                 const Pose2D &pose)
    : id(id), kf_u(0.0, kRDim, kQDim, kRDim), kf_v(0.0, kRDim, kQDim, kRDim),
      kf_yaw(std::isfinite(det.yaw_prior) ? det.yaw_prior : det.rect.yaw,
             kP0Yaw, kQYaw, kRYaw),
      cv_x(0.0, kP0Pos, kP0Vel, kQVel), cv_y(0.0, kP0Pos, kP0Vel, kQVel) {
  if (std::isfinite(det.yaw_prior)) {
    update_yaw(det.rect.yaw);
  }
  const ShapeMeasurement m = measure_shape(det, pose, kf_yaw.x);
  if (m.u_measured) {
    update_extent(&kf_u, &u_measured, m.u_ext);
  }
  if (m.v_measured) {
    update_extent(&kf_v, &v_measured, m.v_ext);
  }
  const Measurement2D z =
      position_measurement(det, pose, m, m.axis.along_u ? m.u_ext : m.v_ext);
  cv_x.pos = z.x;
  cv_y.pos = z.y;
}

// Advance this track's shape and motion filters by one control period.
void TrackedObstacle::predict(double dt_s) {
  kf_u.predict();
  kf_v.predict();
  kf_yaw.predict();
  cv_x.predict(dt_s);
  cv_y.predict(dt_s);
}

// Refresh this track's shape, motion, and yaw from a matched detection.
void TrackedObstacle::update(const RawDetection &det, const Pose2D &pose) {
  const ShapeMeasurement m = measure_shape(det, pose, kf_yaw.x);
  if (m.u_measured) {
    update_extent(&kf_u, &u_measured, m.u_ext);
  }
  if (m.v_measured) {
    update_extent(&kf_v, &v_measured, m.v_ext);
  }

  const Measurement2D z =
      position_measurement(det, pose, m, m.axis.along_u ? u() : v());
  const double rx = z.x - pose.x;
  const double ry = z.y - pose.y;
  const double range = std::hypot(rx, ry);
  const double ux = range > kMinRangeM ? rx / range : 1.0;
  const double uy = range > kMinRangeM ? ry / range : 0.0;
  const double zx = z.x;
  const double zy = z.y;
  const double sig_al = kSigmaAlongM;
  const double sig_ac = std::hypot(kSigmaAcrossM, kSigmaHeadingRad * range);
  const double var_al = sig_al * sig_al;
  const double var_ac = sig_ac * sig_ac;

  refresh_onset(*this, zx - cv_x.pos, zy - cv_y.pos, ux, uy, sig_al, sig_ac,
                range);

  cv_x.update(zx, var_al * ux * ux + var_ac * uy * uy);
  cv_y.update(zy, var_al * uy * uy + var_ac * ux * ux);
  if (!moving && range >= kVelocityRangeMinM) {
    moving =
        cv_x.vel * cv_x.vel / cv_x.P_vel + cv_y.vel * cv_y.vel / cv_y.P_vel >
        kVelocityChi2;
  }

  update_yaw(det.rect.yaw);
  age++;
  missing_count = 0;
}

// Fuse one fitted yaw, taken on the quarter turn nearest the estimate.
void TrackedObstacle::update_yaw(double yaw_fit) {
  kf_yaw.update(kf_yaw.x + wrap_quarter_turn(yaw_fit - kf_yaw.x));
}

// Construct an empty obstacle tracker.
ObstacleTracker::ObstacleTracker() {}

// Predict all tracks and mark them missing until matched.
void ObstacleTracker::predict_tracks() {
  for (TrackedObstacle &obs : m_obstacles) {
    obs.missing_count++;
    obs.predict(kControlPeriodS);
  }
}

// Greedily assign detections to unclaimed tracks within the position gate.
std::vector<int>
ObstacleTracker::associate_detections(const std::vector<RawDetection> &items) {
  const std::size_t n_det = items.size();
  std::vector<int> assignment(n_det, -1);
  std::vector<bool> obs_claimed(m_obstacles.size(), false);

  for (std::size_t i = 0; i < n_det; ++i) {
    const Rect &meas = items[i].rect;
    double best_mahal = kChi2Gate;
    int best_idx = -1;
    for (std::size_t j = 0; j < m_obstacles.size(); ++j) {
      if (obs_claimed[j]) {
        continue;
      }
      const double dx = m_obstacles[j].x() - meas.cx;
      const double dy = m_obstacles[j].y() - meas.cy;
      const double S_x = m_obstacles[j].cv_x.P_pos + TrackedObstacle::kRPos;
      const double S_y = m_obstacles[j].cv_y.P_pos + TrackedObstacle::kRPos;
      const double mahal_sq = (dx * dx) / S_x + (dy * dy) / S_y;
      if (mahal_sq < best_mahal) {
        best_mahal = mahal_sq;
        best_idx = static_cast<int>(j);
      }
    }
    if (best_idx != -1) {
      assignment[i] = best_idx;
      obs_claimed[static_cast<std::size_t>(best_idx)] = true;
    }
  }
  return assignment;
}

// Update matched tracks and create tracks for unmatched detections.
void ObstacleTracker::apply_matches(const std::vector<RawDetection> &items,
                                    const std::vector<int> &assignment,
                                    const Pose2D &pose) {
  for (std::size_t i = 0; i < items.size(); ++i) {
    const RawDetection &det = items[i];
    if (assignment[i] != -1) {
      m_obstacles[static_cast<std::size_t>(assignment[i])].update(det, pose);
    } else {
      m_obstacles.emplace_back(m_next_id++, det, pose);
    }
  }
}

// Remove tracks that exceeded the allowed missed-frame count.
void ObstacleTracker::prune_missing() {
  m_obstacles.erase(std::remove_if(m_obstacles.begin(), m_obstacles.end(),
                                   [](const TrackedObstacle &o) {
                                     return o.missing_count > kMaxMissingFrames;
                                   }),
                    m_obstacles.end());
}

// Convert sufficiently mature tracks into filter obstacle rectangles.
std::vector<Rect> ObstacleTracker::emit_confirmed() const {
  std::vector<Rect> output;
  for (const TrackedObstacle &obs : m_obstacles) {
    if (obs.age < kMinConfirmAge) {
      continue;
    }
    Rect rect;
    rect.cx = obs.x();
    rect.cy = obs.y();
    rect.length = obs.u();
    rect.width = obs.v();
    rect.yaw = obs.yaw();
    rect.vx = obs.vx();
    rect.vy = obs.vy();
    output.push_back(rect);
  }
  return output;
}

// Run one predict-associate-update cycle and emit confirmed tracks.
std::vector<Rect> ObstacleTracker::update(const RawDetections &raw_detections) {
  predict_tracks();
  const std::vector<RawDetection> empty;
  const std::vector<RawDetection> &items =
      raw_detections.valid ? raw_detections.items : empty;
  const std::vector<int> assignment = associate_detections(items);
  apply_matches(items, assignment, raw_detections.pose);
  prune_missing();
  return emit_confirmed();
}

// Discard all tracks and restart obstacle identifiers.
void ObstacleTracker::reset() {
  m_obstacles.clear();
  m_next_id = 0;
}

} // namespace sf
