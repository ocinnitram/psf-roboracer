// Keep safe RL commands or replace them with the closest safe alternative.
#include "sf_filter.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "sf_barrier.h"
#include "sf_geometry.h"
#include "sf_qp.h"

namespace sf {

namespace {

constexpr double kHalfWidthM = 0.5 * kEgoWidthM;
constexpr double kHalfLengthM = 0.5 * kEgoLengthM;
// Sample the track every half body length and retain two nearby obstacles.
constexpr double kStationStepM = kHalfLengthM;
constexpr double kLatchRadiusM = kEgoLengthM;
constexpr std::size_t kMaxObstacles = 2;
constexpr const char *kConstraintName[kConstraintCount] = {
    "obstacle", "wall", "terminal_speed", "terminal_lateral_stop",
    "terminal_cone", "progress"};
// Body corners in the body frame (forward, left).
constexpr double kCornerForwardM[4] = {kHalfLengthM, kHalfLengthM,
                                       -kHalfLengthM, -kHalfLengthM};
constexpr double kCornerLeftM[4] = {kHalfWidthM, -kHalfWidthM, kHalfWidthM,
                                    -kHalfWidthM};

// Squared deviation between two commands in QP units.
double deviation(const Input &u, const Input &reference) {
  const double d0 =
      (u.delta_t - reference.delta_t) / SafetyParameters::qp_steer_scale_rad;
  const double d1 =
      (u.v_des - reference.v_des) / SafetyParameters::qp_speed_scale_mps;
  return d0 * d0 + d1 * d1;
}

// Return the pass side opposite to the supplied side.
PassSide opposite_side(PassSide side) {
  return side == PassSide::Left    ? PassSide::Right
         : side == PassSide::Right ? PassSide::Left
                                   : PassSide::None;
}

// Preview distance of the backup's lateral reference.
double look_ahead_m(double v) {
  return std::max(kEgoLengthM, SafetyParameters::look_s * v);
}

// World point at station s, offset n to the left of the racing line.
Point raceline_point(const TrackHandler &track, double s, double n) {
  const TrackData &d = track.data();
  Point p;
  p.x = track.sample_at_s(d.x, s) + n * track.sample_at_s(d.n_x, s);
  p.y = track.sample_at_s(d.y, s) + n * track.sample_at_s(d.n_y, s);
  return p;
}

} // namespace

// Frenet frame of the body centre.
struct Filter::TrackPose {
  double s = 0.0;
  double n = 0.0;
  double alpha = 0.0; // body yaw relative to the track tangent [rad]
  double yaw_t = 0.0; // track tangent heading, world frame     [rad]
};

// Obstacle position, bounds, and world rectangle at one rollout time.
struct Filter::PredictedObstacle {
  Rect rect;
  double s = 0.0;
  double n_lo = 0.0;
  double n_hi = 0.0;
  double edge_left = 0.0;
  double edge_right = 0.0;
};

// Obstacle motion, bounds, pass room, and selected side in track coordinates.
struct Filter::Obstacle {
  const TrackHandler *track = nullptr;
  Rect rect; // velocity zeroed when accepted as static
  bool moving = false;
  double s = 0.0;
  double n = 0.0;    // centre offset along the track normal      [m]
  double n_lo = 0.0; // lateral extent along the track normal     [m]
  double n_hi = 0.0;
  double v_s = 0.0; // velocity along the tangent and the normal [m/s]
  double v_n = 0.0;
  Point origin = {0.0, 0.0}; // racing-line point at s, offset n  [m]
  double ds = 0.0;           // arc distance ahead of the car now  [m]
  double dodge = 0.0;        // lateral displacement a dodge needs [m]
  double edge_left = 0.0;
  double edge_right = 0.0;
  double wall_left = 0.0;
  double wall_right = 0.0;
  PassSide side = PassSide::None;
  bool engaged = false;

  PredictedObstacle predict(double elapsed_s) const;
};

// Move the obstacle at constant track-frame velocity without changing shape.
Filter::PredictedObstacle Filter::Obstacle::predict(double elapsed_s) const {
  const double shift_n = v_n * elapsed_s;
  PredictedObstacle o;
  o.s = s + v_s * elapsed_s;
  o.n_lo = n_lo + shift_n;
  o.n_hi = n_hi + shift_n;
  o.edge_left = edge_left + shift_n;
  o.edge_right = edge_right + shift_n;
  const Point moved = raceline_point(*track, o.s, n + shift_n);
  o.rect = rect;
  o.rect.cx += moved.x - origin.x;
  o.rect.cy += moved.y - origin.y;
  return o;
}

// Minimum safety-constraint values and predicted states over one rollout.
struct Filter::Rollout {
  double constraint_values[kConstraintCount];
  int stages = 0;
  std::vector<State> states;
  // Command entering each stage; recorded only with Request::export_commands.
  std::vector<Input> commands;

  Rollout() {
    for (double &value : constraint_values) {
      value = kInf;
    }
  }
  // Return the smallest safety value, excluding relative progress.
  double minimum_safety_value() const {
    double worst = kInf;
    for (int i = 0; i < kConstraintProgress; ++i) {
      worst = std::min(worst, constraint_values[i]);
    }
    return worst;
  }
  bool is_feasible() const { return minimum_safety_value() >= 0.0; }
  double minimum_clearance() const {
    return std::min(constraint_values[kConstraintObstacle],
                    constraint_values[kConstraintWall]);
  }
  const char *worst_constraint_name() const {
    int worst = 0;
    for (int i = 1; i < kConstraintProgress; ++i) {
      if (constraint_values[i] < constraint_values[worst]) {
        worst = i;
      }
    }
    return constraint_values[worst] >= 0.0 ? "ok"
                                           : kConstraintName[worst];
  }
};

// A command and its evaluated rollout under one backup option.
struct Filter::EvaluatedCommand {
  Input u;
  Rollout rollout;
  Outcome outcome = Outcome::Uncertified;
  BackupOption option;
  // Changed commands must satisfy progress; the original RL command need not.
  bool satisfies_requirements(
      const double required_values[kConstraintCount], bool liveness) const {
    const int count = liveness ? kConstraintCount : kConstraintProgress;
    for (int i = 0; i < count; ++i) {
      if (rollout.constraint_values[i] < required_values[i]) {
        return false;
      }
    }
    return true;
  }
};

// Backup and RL evaluations for one pass-side or stop option.
struct Filter::BackupEvaluation {
  BackupOption option;
  EvaluatedCommand backup;
  EvaluatedCommand rl;
  double required_values[kConstraintCount];
  bool is_robust = false;
};

// Selected command plus the RL rollout used for diagnostics.
struct Filter::CommandSearchResult {
  EvaluatedCommand published;
  EvaluatedCommand rl;
};

// Create a filter for one track.
Filter::Filter(const TrackHandler *track, double v_max_mps)
    : m_track(track), m_limits(v_max_mps) {}

// Clear remembered pass-side and prediction state.
void Filter::reset() {
  m_remembered_pass_side = RememberedPassSide{};
  m_have_predicted = false;
}

// Convert a vehicle state to its local track frame.
Filter::TrackPose Filter::track_pose(const State &z) const {
  const Frenet f = m_track->to_frenet(z.x, z.y, z.psi);
  TrackPose pose;
  pose.s = f.s;
  pose.n = f.n;
  pose.alpha = f.alpha;
  pose.yaw_t = wrap_pi(z.psi - f.alpha);
  return pose;
}

// Return the signed shortest track distance between two stations.
double Filter::ahead_m(double s_from, double s_to) const {
  const double length = m_track->length();
  double d = std::fmod(s_to - s_from + length, length);
  if (d > 0.5 * length) {
    d -= length;
  }
  return d;
}

// ---- Obstacles --------------------------------------------------------------

// Convert a relevant tracked rectangle to track coordinates.
bool Filter::obstacle_in_track_frame(const Rect &rect, const TrackPose &now,
                                     Obstacle *obstacle) const {
  const Frenet f = m_track->to_frenet(rect.cx, rect.cy, 0.0);
  const double yaw_t = wrap_pi(0.0 - f.alpha);
  const double tx = std::cos(yaw_t);
  const double ty = std::sin(yaw_t);
  const double extent = rect_half_extent(rect, -ty, tx);
  obstacle->track = m_track;
  obstacle->rect = rect;
  obstacle->moving = std::hypot(rect.vx, rect.vy) >=
                     SafetyParameters::moving_speed_floor_mps;
  if (!obstacle->moving) {
    obstacle->rect.vx = 0.0;
    obstacle->rect.vy = 0.0;
  }
  obstacle->s = f.s;
  obstacle->n = f.n;
  obstacle->n_lo = f.n - extent;
  obstacle->n_hi = f.n + extent;
  obstacle->v_s = obstacle->rect.vx * tx + obstacle->rect.vy * ty;
  obstacle->v_n = -obstacle->rect.vx * ty + obstacle->rect.vy * tx;
  obstacle->origin = raceline_point(*m_track, f.s, f.n);
  obstacle->ds = ahead_m(now.s, obstacle->s);
  const bool off_track = obstacle->n_lo > m_track->width_left(obstacle->s) ||
                         obstacle->n_hi < -m_track->width_right(obstacle->s);
  const bool behind = obstacle->ds < -(kEgoLengthM + kHalfLengthM + extent);
  return !off_track && !behind;
}

// Measure collision-free room on each side of an obstacle.
void Filter::measure_pass_room(Obstacle *obstacle) const {
  const double disk =
      cover_disk_radius() + SafetyParameters::obstacle_margin_m;
  const double corner = kHalfWidthM + SafetyParameters::wall_margin_m;
  obstacle->dodge = 0.5 * (obstacle->n_hi - obstacle->n_lo) + disk;
  obstacle->wall_left = kInf;
  obstacle->wall_right = -kInf;
  for (double ds = -kEgoLengthM; ds <= kEgoLengthM; ds += kStationStepM) {
    obstacle->wall_left = std::min(
        obstacle->wall_left, m_track->width_left(obstacle->s + ds) - corner);
    obstacle->wall_right = std::max(
        obstacle->wall_right, -m_track->width_right(obstacle->s + ds) + corner);
  }
  obstacle->edge_left = obstacle->n_hi + disk;
  obstacle->edge_right = obstacle->n_lo - disk;
}

// Keep a usable remembered pass side or select the roomier side.
PassSide Filter::choose_side(const Obstacle &obstacle, bool holds_latch) const {
  const double gap_left = obstacle.wall_left - obstacle.edge_left;
  const double gap_right = obstacle.edge_right - obstacle.wall_right;
  PassSide side = PassSide::None;
  if (holds_latch &&
      std::hypot(m_remembered_pass_side.x - obstacle.rect.cx,
                 m_remembered_pass_side.y - obstacle.rect.cy) < kLatchRadiusM) {
    side = m_remembered_pass_side.side;
  }
  if ((side == PassSide::Left && gap_left < 0.0) ||
      (side == PassSide::Right && gap_right < 0.0)) {
    side = PassSide::None;
  }
  if (side == PassSide::None) {
    side = (gap_left >= gap_right) ? PassSide::Left : PassSide::Right;
  }
  return side;
}

// Prepare and order nearby obstacles for the safety search.
std::vector<Filter::Obstacle> Filter::admit_obstacles(const Request &request,
                                                      const TrackPose &now) {
  std::vector<Obstacle> obstacles;
  for (const Rect &rect : request.obstacles) {
    Obstacle obstacle;
    if (obstacle_in_track_frame(rect, now, &obstacle)) {
      obstacles.push_back(obstacle);
    }
  }
  // Ahead first by increasing ds, then just-passed by increasing |ds|, so a
  // crate still inside the behind window cannot displace one further ahead.
  std::sort(obstacles.begin(), obstacles.end(),
            [](const Obstacle &a, const Obstacle &b) {
              const bool a_ahead = a.ds >= 0.0;
              const bool b_ahead = b.ds >= 0.0;
              if (a_ahead != b_ahead) {
                return a_ahead;
              }
              return std::abs(a.ds) < std::abs(b.ds);
            });
  if (obstacles.size() > kMaxObstacles) {
    obstacles.resize(kMaxObstacles);
  }
  const double v = std::max(request.z.v, kKinematicFloorMps);
  bool nearest_engaged = true;
  for (Obstacle &obstacle : obstacles) {
    measure_pass_room(&obstacle);
    const double engage_m =
        escape_distance(v, obstacle.dodge,
                        SafetyParameters::lat_budget_mps2) +
        SafetyParameters::settle_s * v;
    obstacle.engaged = obstacle.ds > -kEgoLengthM && obstacle.ds < engage_m;
    const bool holds_latch = obstacle.engaged && nearest_engaged;
    obstacle.side = choose_side(obstacle, holds_latch);
    if (holds_latch) {
      m_remembered_pass_side = {
          obstacle.rect.cx, obstacle.rect.cy, obstacle.side};
      nearest_engaged = false;
    }
  }
  return obstacles;
}

// ---- Backup policy -------------------------------------------------------

// Choose a safe lateral target from the track and obstacle gaps ahead.
double Filter::reference_offset(const TrackPose &car, double car_v,
                                const std::vector<Obstacle> &obstacles,
                                const BackupOption &option,
                                double elapsed_s) const {
  const double inset = kHalfWidthM + SafetyParameters::wall_margin_m;
  const double s_query = car.s + look_ahead_m(car_v);
  double lo = -kInf;
  double hi = kInf;
  const double span = std::max(ahead_m(car.s, s_query), brake_reach_m(car_v));
  for (double ds = 0.0; ds <= span + 1e-9; ds += kStationStepM) {
    lo = std::max(lo, -m_track->width_right(car.s + ds) + inset);
    hi = std::min(hi, m_track->width_left(car.s + ds) - inset);
  }
  bool nearest = true; // the option applies to the nearest engaged obstacle
  for (const Obstacle &c : obstacles) {
    if (!c.engaged) {
      continue;
    }
    const PredictedObstacle o = c.predict(elapsed_s);
    if (ahead_m(car.s, o.s) <= -kEgoLengthM) {
      continue;
    }
    if (ahead_m(s_query, o.s) > -(kEgoLengthM + kHalfLengthM)) {
      const PassSide side = (nearest && option.side != PassSide::None)
                                ? option.side
                                : c.side;
      lo = std::max(lo, (side == PassSide::Left) ? o.edge_left : c.wall_right);
      hi = std::min(hi, (side == PassSide::Left) ? c.wall_left : o.edge_right);
      break;
    }
    nearest = false;
  }
  if (hi - lo <= 2.0 * m_limits.reference_inset_m) {
    return 0.5 * (lo + hi);
  }
  return std::max(lo + m_limits.reference_inset_m,
                  std::min(hi - m_limits.reference_inset_m, car.n));
}

// Compute braking distance at the pass rate including control delay.
double Filter::brake_reach_m(double v) const {
  return v * v / (2.0 * SafetyParameters::brake_pass_mps2) +
         v * (speed_switch_s() + kTauAccelS);
}

// Compute a speed that can meet each curvature limit ahead.
double Filter::corridor_speed(double s, double v) const {
  const double reach = brake_reach_m(v);
  double v_safe = m_limits.v_max;
  for (double ds = 0.0; ds <= reach + kStationStepM; ds += kStationStepM) {
    const double v_lim =
        curvature_speed_limit(m_track->kappa_s(s + ds),
                              SafetyParameters::lat_budget_mps2,
                              m_limits.v_max);
    v_safe = std::min(
        v_safe,
        std::sqrt(v_lim * v_lim +
                  2.0 * SafetyParameters::brake_pass_mps2 * ds));
  }
  return v_safe;
}

// Braking the friction circle leaves after the lateral demand at a state:
// the larger of what the car is doing (v r) and the lateral acceleration
// `a_y_mps2` that is asked of it. Lateral has priority, so a car cornering at
// the budget may not brake at all; braking there is what lets the rear axle
// go.
double Filter::brake_room_mps2(const State &z, double a_y_mps2) const {
  const double v = std::max(z.v, kKinematicFloorMps);
  const double a_y = std::max(std::abs(v * z.r), std::abs(a_y_mps2));
  return brake_room(a_y, SafetyParameters::lat_budget_mps2);
}

// Lowest speed setpoint whose braking demand stays inside the braking room
// left by what the car's station asks (kappa v^2).
double Filter::speed_floor_mps(const State &z) const {
  const double v = std::max(z.v, kKinematicFloorMps);
  const double kappa_v2 = m_track->kappa_s(track_pose(z).s) * v * v;
  return std::max(0.0, z.v - kTauSpeedPlantS * brake_room_mps2(z, kappa_v2));
}

// Generate delayed backup steering and speed for one pass or stop option.
Input Filter::backup_command(const State &z_now, const Input &before,
                             const Input &active,
                             const std::vector<Obstacle> &obstacles,
                             const BackupOption &option, double now_s) const {
  // Same three-command period as the certificate, then the leftover
  // (steer_switch - T) still under u_prev.
  State z = sf::step(z_now, before, active, active, kControlPeriodS,
                     m_limits.v_max);
  const double remain_s = steer_switch_s() - kControlPeriodS;
  if (remain_s > 0.0) {
    z = sf::step(z, active, active, active, remain_s, m_limits.v_max);
  }
  const TrackPose f = track_pose(z);
  const double v = std::max(z.v, kKinematicFloorMps);

  // Lateral demand toward the safe offset.
  const double theta = wrap_pi(f.alpha + z.beta);
  const double n_dot = z.v * std::sin(theta);
  const double e =
      reference_offset(f, v, obstacles, option, now_s + steer_switch_s()) - f.n;
  const double n_dot_des = std::copysign(
      std::min(std::abs(e) / SafetyParameters::time_s,
               SafetyParameters::max_crossing_sin * v),
      e);
  const double n_ddot =
      (n_dot_des - n_dot) / SafetyParameters::lateral_tau_s;
  const double a_y_demand = m_track->kappa_s(f.s) * v * v + n_ddot;

  // Approach cruise or rest within braking and corridor limits. Lateral has
  // priority for the curvature's share of the demand, kappa v^2, and no more
  // than the demand itself: a car already turned into the corner asks less and
  // may brake with the rest. Reaching a new offset yields to braking.
  const double v_target = option.stop ? 0.0 : SafetyParameters::cruise_mps;
  const double brake =
      option.stop ? kMaxAccelMps2 : SafetyParameters::brake_pass_mps2;
  double a_x = std::max(
      -brake,
      std::min(kMaxAccelMps2,
               (v_target - z.v) / SafetyParameters::time_s));
  a_x = std::max(-kMaxAccelMps2,
                 std::min(a_x, (corridor_speed(f.s, v) - v) /
                                   SafetyParameters::time_s));
  a_x = std::max(
      a_x, -brake_room_mps2(z, std::min(std::abs(m_track->kappa_s(f.s)) * v * v,
                                        std::abs(a_y_demand))));

  // Steer toward the safe offset within the remaining tyre grip.
  const double a_y_cap =
      std::min(SafetyParameters::lat_budget_mps2,
               SafetyParameters::grip_eta * rear_lateral_accel_limit(a_x));
  const double a_y = std::max(-a_y_cap, std::min(a_y_cap, a_y_demand));
  const double delta_cap = delta_max_rad(v, a_y_cap);
  Input u;
  u.delta_t = std::max(-delta_cap, std::min(delta_cap, delta_for_ay(a_y, v)));
  u.v_des = std::max(0.0, z.v + kTauSpeedPlantS * a_x);
  return u;
}

// ---- Rollout and rows ----------------------------------------------------

// Update wall and obstacle clearances at one rollout stage.
void Filter::update_stage_constraints(
    const State &z, const State &prev, const TrackPose &f,
    const std::vector<Obstacle> &obstacles, double elapsed_s,
    double values[kConstraintCount]) const {
  const double tx = std::cos(f.yaw_t);
  const double ty = std::sin(f.yaw_t);
  const Rotation heading = rotation(z.psi);
  const Rotation heading_prev = rotation(prev.psi);
  for (int k = 0; k < 4; ++k) {
    const Point p =
        body_point(z, heading, kCornerForwardM[k], kCornerLeftM[k]);
    const double dx = p.x - z.x;
    const double dy = p.y - z.y;
    const double s_k = f.s + dx * tx + dy * ty;
    const double n_k = f.n - dx * ty + dy * tx;
    const double room = std::min(m_track->width_left(s_k) - n_k,
                                 n_k + m_track->width_right(s_k));
    values[kConstraintWall] =
        std::min(values[kConstraintWall],
                 room - SafetyParameters::wall_margin_m);
  }
  // Check moving obstacles at both predicted and current positions.
  const double keep_out_m =
      cover_disk_radius() + SafetyParameters::obstacle_margin_m;
  const double budget_m =
      2.0 * SafetyParameters::obstacle_vel_sigma_mps * elapsed_s;
  for (const Obstacle &c : obstacles) {
    const int hypotheses = c.moving ? 2 : 1;
    for (int h = 0; h < hypotheses; ++h) {
      const bool predicted = c.moving && h == 0;
      const Rect rect = predicted ? c.predict(elapsed_s).rect : c.rect;
      const double keep = keep_out_m + (predicted ? budget_m : 0.0);
      const Rotation rect_yaw = rotation(rect.yaw);
      for (int i = 0; i < SafetyParameters::cover_disks; ++i) {
        const Point q1 = cover_disk_centre(z, heading, i);
        const Point q0 = cover_disk_centre(prev, heading_prev, i);
        const double d_end = rect_signed_distance(rect, rect_yaw, q1.x, q1.y);
        const double d_mid =
            rect_signed_distance(rect, rect_yaw, 0.5 * (q0.x + q1.x),
                                 0.5 * (q0.y + q1.y));
        values[kConstraintObstacle] = std::min(
            values[kConstraintObstacle], std::min(d_end, d_mid) - keep);
      }
    }
  }
}

// Evaluate speed, heading, and obstacle conditions at the rollout end.
void Filter::evaluate_terminal_constraints(
    const State &z_end, const TrackPose &f_end,
    const std::vector<Obstacle> &obstacles, double elapsed_s,
    double values[kConstraintCount]) const {
  const double v_end = std::max(z_end.v, kKinematicFloorMps);
  values[kConstraintSpeed] = corridor_speed(f_end.s, z_end.v) - z_end.v;

  // The lateral-stop row uses the heading relative to the track without the
  // sideslip: the carried sideslip is integrated open loop with the nominal
  // tyre and is unreliable under a different tyre.
  const double theta_end = wrap_pi(f_end.alpha);
  const double a_perp = SafetyParameters::lat_budget_mps2 -
                        std::abs(m_track->kappa_s(f_end.s)) * v_end * v_end;
  // The body's extent across the track follows where it points (alpha), not
  // where the COG moves (alpha + beta).
  const double half_extent_m =
      0.5 * (kEgoLengthM * std::abs(std::sin(f_end.alpha)) +
             kEgoWidthM * std::abs(std::cos(f_end.alpha)));
  const double room_end =
      std::max(0.0, std::min(m_track->width_left(f_end.s) - f_end.n,
                             m_track->width_right(f_end.s) + f_end.n) -
                        half_extent_m);
  values[kConstraintLateralStop] =
      lateral_stop_heading_limit(a_perp, room_end, v_end) -
      std::abs(theta_end);

  // The rolled-out state carries the actual sideslip and yaw rate; the
  // steady-state closure is not needed here.
  SteadyStateCornering c_end{};
  c_end.beta = z_end.beta;
  c_end.r = z_end.r;
  const double v_end_floor = std::max(z_end.v, kKinematicFloorMps);
  const double ego_vx = v_end_floor * std::cos(z_end.psi + z_end.beta);
  const double ego_vy = v_end_floor * std::sin(z_end.psi + z_end.beta);
  const double disk_keep =
      cover_disk_radius() + SafetyParameters::obstacle_margin_m;
  const Rotation heading_end = rotation(z_end.psi);
  const Rotation slip_end = rotation(z_end.beta);
  for (const Obstacle &c : obstacles) {
    // Check both predicted motion and the current static position.
    const int hypotheses = c.moving ? 2 : 1;
    for (int h = 0; h < hypotheses; ++h) {
      const bool predicted = c.moving && h == 0;
      const PredictedObstacle o = c.predict(predicted ? elapsed_s : 0.0);
      const double obs_vx = predicted ? o.rect.vx : 0.0;
      const double obs_vy = predicted ? o.rect.vy : 0.0;
      const double half_diagonal =
          0.5 * std::hypot(o.rect.width, o.rect.length);
      if (ahead_m(f_end.s, o.s) <= kHalfLengthM + half_diagonal) {
        continue;
      }
      const double d = std::hypot(o.rect.cx - z_end.x, o.rect.cy - z_end.y);
      // Use relative speed to decide whether the obstacle is still reachable.
      const double v_rel = std::hypot(ego_vx - obs_vx, ego_vy - obs_vy);
      if (d >= escape_distance(v_rel, c.dodge,
                               SafetyParameters::lat_budget_mps2)) {
        continue;
      }
      for (int i = 0; i < SafetyParameters::cover_disks; ++i) {
        const Point q = cover_disk_centre(z_end, heading_end, i);
        // Relative velocity of each ego cover disk: the body velocity
        // (v cos b, v sin b + l_i r) rotated into the world, minus the
        // obstacle's.
        const double phi_ego =
            disk_velocity_heading(z_end, c_end, cover_disk_offset(i));
        const double speed_i = std::hypot(
            v_end_floor * slip_end.c,
            v_end_floor * slip_end.s + cover_disk_offset(i) * c_end.r);
        const double disk_vx = speed_i * std::cos(phi_ego) - obs_vx;
        const double disk_vy = speed_i * std::sin(phi_ego) - obs_vy;
        const double phi = std::atan2(disk_vy, disk_vx);
        const double hc = cone_h(q.x, q.y, phi, o.rect.cx, o.rect.cy,
                                 disk_keep + half_diagonal);
        values[kConstraintCone] = std::min(values[kConstraintCone], hc);
      }
    }
  }
}

// Simulate a command followed by backup and collect safety values.
Filter::Rollout Filter::predict_constraints(
    const Request &request, const Input &command,
    const std::vector<Obstacle> &obstacles,
    const BackupOption &option) const {
  Rollout out;
  out.states.reserve(m_limits.max_stages + 1);
  out.states.push_back(request.z);
  if (request.export_commands) {
    out.commands.reserve(m_limits.max_stages);
  }
  TrackPose f = track_pose(request.z);
  const double s_start = f.s;
  Input before = request.u_before;
  Input active = request.u_prev;
  Input next = command;
  for (int j = 1; j <= m_limits.max_stages; ++j) {
    if (j > 1) {
      before = active;
      active = next;
      next = backup_command(out.states[j - 1], before, active, obstacles,
                            option, (j - 1) * kControlPeriodS);
    }
    const State &prev = out.states.back();
    const State z =
        sf::step(prev, before, active, next, kControlPeriodS, m_limits.v_max);
    f = track_pose(z);
    const double elapsed_s = j * kControlPeriodS;
    update_stage_constraints(z, prev, f, obstacles, elapsed_s,
                             out.constraint_values);
    out.states.push_back(z);
    if (request.export_commands) {
      out.commands.push_back(next);
    }
    out.stages = j;
    bool pending = false;
    for (const Obstacle &c : obstacles) {
      if (!c.engaged) {
        continue;
      }
      pending =
          pending || (ahead_m(f.s, c.predict(elapsed_s).s) > -kEgoLengthM);
    }
    if (j >= m_limits.min_stages && !pending) {
      break;
    }
  }
  out.constraint_values[kConstraintProgress] = ahead_m(s_start, f.s);
  evaluate_terminal_constraints(out.states.back(), f, obstacles,
                                out.stages * kControlPeriodS,
                                out.constraint_values);
  return out;
}

// ---- Members and the certificate -----------------------------------------

// Evaluate one command under one backup option.
Filter::EvaluatedCommand Filter::evaluate_command(
    const Request &request, const std::vector<Obstacle> &obstacles,
    const BackupOption &option, const Input &u, int *rollouts) const {
  ++*rollouts;
  EvaluatedCommand c;
  c.u = u;
  c.option = option;
  c.rollout = predict_constraints(request, u, obstacles, option);
  return c;
}

// Evaluate the backup and RL commands for one pass or stop option.
Filter::BackupEvaluation Filter::evaluate_backup_option(
    const Request &request, const std::vector<Obstacle> &obstacles,
    const BackupOption &option, int *rollouts) const {
  BackupEvaluation evaluation;
  evaluation.option = option;
  evaluation.backup = evaluate_command(
      request, obstacles, option,
      backup_command(request.z, request.u_before, request.u_prev, obstacles,
                     option, 0.0),
      rollouts);
  evaluation.rl =
      evaluate_command(request, obstacles, option, request.u_rl, rollouts);
  const double margin_obstacle = certify_margin(request.z.v);
  const double margin_wall = certify_wall_margin(request.z.v);
  evaluation.is_robust = true;
  for (int i = 0; i < kConstraintProgress; ++i) {
    const double robust = (i == kConstraintObstacle) ? margin_obstacle
                          : (i == kConstraintWall)   ? margin_wall
                                                    : 0.0;
    evaluation.required_values[i] =
        std::min(robust, evaluation.backup.rollout.constraint_values[i]);
    evaluation.is_robust =
        evaluation.is_robust &&
        evaluation.backup.rollout.constraint_values[i] >= robust;
  }
  evaluation.required_values[kConstraintProgress] =
      evaluation.is_robust
          ? std::min(evaluation.backup.rollout
                             .constraint_values[kConstraintProgress] -
                         m_limits.progress_tol_m,
                     evaluation.rl.rollout
                         .constraint_values[kConstraintProgress])
          : -kInf;
  if (evaluation.rl.satisfies_requirements(evaluation.required_values, false)) {
    evaluation.rl.outcome = evaluation.rl.rollout.is_feasible()
                                ? Outcome::Passthrough
                                : Outcome::Uncertified;
  }
  return evaluation;
}

// Linearize the safety rows and solve one bounded QP correction.
bool Filter::qp_step(const Request &request,
                     const std::vector<Obstacle> &obstacles,
                     const BackupEvaluation &evaluation,
                     const EvaluatedCommand &current, int *rollouts,
                     EvaluatedCommand *next) const {
  const double scale[2] = {SafetyParameters::qp_steer_scale_rad,
                           SafetyParameters::qp_speed_scale_mps};
  const double h_step[2] = {kGradStepSteerRad, kGradStepSpeedMps};
  double grad[kConstraintCount][2];
  for (int c = 0; c < 2; ++c) {
    Input lo = current.u;
    Input hi = current.u;
    (c == 0 ? lo.delta_t : lo.v_des) -= h_step[c];
    (c == 0 ? hi.delta_t : hi.v_des) += h_step[c];
    const Rollout r_lo =
        predict_constraints(request, lo, obstacles, evaluation.option);
    const Rollout r_hi =
        predict_constraints(request, hi, obstacles, evaluation.option);
    *rollouts += 2;
    for (int i = 0; i < kConstraintCount; ++i) {
      const bool finite = r_hi.constraint_values[i] < kInf &&
                          r_lo.constraint_values[i] < kInf;
      grad[i][c] = finite ? (r_hi.constraint_values[i] -
                             r_lo.constraint_values[i]) /
                                (2.0 * h_step[c])
                          : 0.0;
    }
  }
  const double du[2] = {current.u.delta_t - request.u_rl.delta_t,
                        current.u.v_des - request.u_rl.v_des};
  // Build linearized safety constraints in normalized command coordinates.
  std::vector<HalfPlane> constraints;
  for (int i = 0; i < kConstraintCount; ++i) {
    if (current.rollout.constraint_values[i] < kInf &&
        evaluation.required_values[i] > -kInf) {
      // kQpInterior is a small margin for linearisation error.
      const double b = evaluation.required_values[i] + kQpInterior -
                       current.rollout.constraint_values[i] +
                       grad[i][0] * du[0] + grad[i][1] * du[1];
      constraints.push_back(
          {grad[i][0] * scale[0], grad[i][1] * scale[1], b});
    }
  }
  const double reach = kSteerRateMaxRadps * kControlPeriodS;
  const double lo_d =
      std::max(-kSteerLockTargetRad, request.u_prev.delta_t - reach);
  const double hi_d =
      std::min(kSteerLockTargetRad, request.u_prev.delta_t + reach);
  constraints.push_back({-scale[0], 0.0, request.u_rl.delta_t - hi_d});
  constraints.push_back({scale[0], 0.0, lo_d - request.u_rl.delta_t});
  constraints.push_back(
      {0.0, -scale[1], request.u_rl.v_des - m_limits.v_max});
  constraints.push_back(
      {0.0, scale[1], speed_floor_mps(request.z) - request.u_rl.v_des});
  double steering_correction = 0.0;
  double speed_correction = 0.0;
  if (!min_norm_point(
          constraints, &steering_correction, &speed_correction)) {
    return false;
  }
  const Input u = {request.u_rl.delta_t + scale[0] * steering_correction,
                   request.u_rl.v_des + scale[1] * speed_correction};
  *next =
      evaluate_command(request, obstacles, evaluation.option, u, rollouts);
  return true;
}

// Find the closest safe point on the segment toward the backup command.
Filter::EvaluatedCommand Filter::bisect_toward_backup(
    const Request &request, const std::vector<Obstacle> &obstacles,
    const BackupEvaluation &evaluation, const Input &from,
    int *rollouts) const {
  EvaluatedCommand lo = evaluation.backup;
  // The backup honours the braking room; the segment must not leave it.
  Input hi = from;
  hi.v_des = std::max(hi.v_des, speed_floor_mps(request.z));
  for (int i = 0; i < kSegmentHalvings; ++i) {
    const Input mid = {0.5 * (lo.u.delta_t + hi.delta_t),
                       0.5 * (lo.u.v_des + hi.v_des)};
    const EvaluatedCommand c = evaluate_command(
        request, obstacles, evaluation.option, mid, rollouts);
    if (c.satisfies_requirements(evaluation.required_values, true)) {
      lo = c;
    } else {
      hi = c.u;
    }
  }
  const bool is_backup =
      lo.u.delta_t == evaluation.backup.u.delta_t &&
      lo.u.v_des == evaluation.backup.u.v_des;
  lo.outcome = !lo.rollout.is_feasible() ? Outcome::Uncertified
               : is_backup               ? Outcome::Backup
                                         : Outcome::Segment;
  return lo;
}

// Keep the RL command or correct it by QP and then segment search.
Filter::EvaluatedCommand Filter::correct_for_backup_option(
    const Request &request, const std::vector<Obstacle> &obstacles,
    const BackupEvaluation &evaluation, int *rollouts) const {
  if (evaluation.rl.satisfies_requirements(evaluation.required_values,
                                           false)) {
    return evaluation.rl;
  }
  EvaluatedCommand current = evaluation.rl;
  for (int iter = 0; iter < kSqpIters; ++iter) {
    EvaluatedCommand next;
    if (!qp_step(request, obstacles, evaluation, current, rollouts, &next)) {
      break;
    }
    current = next;
    if (current.satisfies_requirements(evaluation.required_values, true)) {
      current.outcome = current.rollout.is_feasible()
                            ? Outcome::Qp
                            : Outcome::Uncertified;
      return current;
    }
  }
  return bisect_toward_backup(request, obstacles, evaluation, current.u,
                              rollouts);
}

// Search remembered-side, opposite-side, and stop options for the best command.
Filter::CommandSearchResult Filter::select_best_command(
    const Request &request, const std::vector<Obstacle> &obstacles,
    const Obstacle *nearest, int *rollouts) const {
  auto is_certified = [](const EvaluatedCommand &c) {
    return c.outcome != Outcome::Uncertified;
  };
  auto is_better = [&](const EvaluatedCommand &c,
                       const EvaluatedCommand &b) {
    if (is_certified(c) != is_certified(b)) {
      return is_certified(c);
    }
    if (is_certified(c)) {
      return deviation(c.u, request.u_rl) < deviation(b.u, request.u_rl);
    }
    return c.rollout.minimum_safety_value() >
           b.rollout.minimum_safety_value();
  };
  BackupEvaluation remembered = evaluate_backup_option(
      request, obstacles, BackupOption{}, rollouts);
  EvaluatedCommand best = remembered.rl;
  if (best.outcome == Outcome::Passthrough) {
    return {std::move(best), std::move(remembered.rl)};
  }
  BackupEvaluation other;
  if (nearest) {
    other = evaluate_backup_option(
        request, obstacles, BackupOption{opposite_side(nearest->side), false},
        rollouts);
    if (other.rl.outcome == Outcome::Passthrough && other.is_robust) {
      return {std::move(other.rl), std::move(remembered.rl)};
    }
  }
  best =
      correct_for_backup_option(request, obstacles, remembered, rollouts);
  // When the policy steers toward the other side than the backup does,
  // correct inside that side too, so a certified correction on the side the
  // policy is heading for can win over one across it.
  const double lean = request.u_rl.delta_t - remembered.backup.u.delta_t;
  const bool leans_other =
      nearest && (other.option.side == PassSide::Left ? lean > 0.0 : lean < 0.0);
  if ((leans_other || !is_certified(best)) && nearest) {
    const EvaluatedCommand c =
        correct_for_backup_option(request, obstacles, other, rollouts);
    best = is_better(c, best) ? c : best;
  }
  if (!is_certified(best)) {
    const BackupEvaluation rest = evaluate_backup_option(
        request, obstacles, BackupOption{PassSide::None, true}, rollouts);
    const EvaluatedCommand c =
        correct_for_backup_option(request, obstacles, rest, rollouts);
    best = is_better(c, best) ? c : best;
  }
  return {std::move(best), std::move(remembered.rl)};
}

// ---- One tick ------------------------------------------------------------

// Select and publish the safest command for one control period.
Result Filter::step(const Request &request_in) {
  Request request = request_in;
  if (m_have_predicted) {
    request.z.beta = m_predicted.beta;
    request.z.fy_f = m_predicted.fy_f;
    request.z.fy_r = m_predicted.fy_r;
    request.z.delta_dot = m_predicted.delta_dot;
    request.z.accel = m_predicted.accel;
  }
  const std::vector<Obstacle> obstacles =
      admit_obstacles(request, track_pose(request.z));
  // Only the nearest engaged obstacle can change the selected pass side.
  const Obstacle *nearest = nullptr;
  for (const Obstacle &c : obstacles) {
    if (c.engaged && !nearest) {
      nearest = &c;
    }
  }

  Result result;
  CommandSearchResult search =
      select_best_command(request, obstacles, nearest, &result.rollouts);
  EvaluatedCommand &best = search.published;
  result.engaged = nearest != nullptr;
  result.side = nearest ? nearest->side : PassSide::None;
  if (best.option.side != PassSide::None) {
    result.side = best.option.side;
    m_remembered_pass_side.side = best.option.side;
  }
  for (int i = 0; i < kConstraintCount; ++i) {
    result.constraint_values_rl[i] = search.rl.rollout.constraint_values[i];
    result.constraint_values_published[i] =
        best.rollout.constraint_values[i];
  }
  if (best.rollout.states.size() > 1) {
    m_predicted = best.rollout.states[1];
    m_have_predicted = true;
  }
  result.u = best.u;
  result.outcome = best.outcome;
  result.stop = best.option.stop;
  result.minimum_clearance_rl = search.rl.rollout.minimum_clearance();
  result.minimum_clearance_published = best.rollout.minimum_clearance();
  result.stages = best.rollout.stages;
  result.reason = best.rollout.worst_constraint_name();
  result.rollout = std::move(best.rollout.states);
  result.commands = std::move(best.rollout.commands);
  return result;
}

} // namespace sf
