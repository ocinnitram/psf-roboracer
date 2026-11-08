// Test filter behavior on clear track and static or moving obstacles.

#include "sf_filter.h"
#include "sf_barrier.h"
#include "sf_math.h"
#include "sf_tuning.h"
#include "vehicle/params.h"

#include <cmath>
#include <cstdio>

using namespace sf;

namespace {

constexpr const char* kTrackPath = "tracks/lab-20260906/raceline.csv";
// Top speed of the lab vehicle.
constexpr double kVMaxMps = 4.0;
constexpr double kCrateLengthM = 0.35;
constexpr double kCrateWidthM = 0.25;
constexpr double kWaitGapM = 0.10;
constexpr double kCrateSpeedMps = 0.5;
int failures = 0;

void check(bool condition, const char* message) {
    std::printf("  [%s] %s\n", condition ? "PASS" : "FAIL", message);
    if (!condition) {
        ++failures;
    }
}

bool near(double value, double expected, double tolerance) {
    return std::abs(value - expected) <= tolerance;
}

State predict_backup_start(const State& z, const Input& before, const Input& prev) {
    State predicted = step(z, before, prev, prev, kControlPeriodS, kVMaxMps);
    const double remain_s = steer_switch_s() - kControlPeriodS;
    if (remain_s > 0.0) {
        predicted = step(predicted, prev, prev, prev, remain_s, kVMaxMps);
    }
    return predicted;
}

State pose_at(const TrackHandler& track, int idx, double n, double v) {
    const TrackData& d = track.data();
    State z;
    z.x = d.x[idx] + n * d.n_x[idx];
    z.y = d.y[idx] + n * d.n_y[idx];
    z.psi = d.track_yaw[idx];
    z.v = v;
    return z;
}

int station_at_s(const TrackHandler& track, double s) {
    const TrackData& d = track.data();
    int best = 0;
    for (int i = 0; i < static_cast<int>(d.s.size()); ++i) {
        if (std::abs(d.s[i] - s) < std::abs(d.s[best] - s)) {
            best = i;
        }
    }
    return best;
}

Rect obstacle_at(const TrackHandler& track, double s, double n) {
    const TrackData& d = track.data();
    const int idx = station_at_s(track, s);
    Rect r;
    r.cx = d.x[idx] + n * d.n_x[idx];
    r.cy = d.y[idx] + n * d.n_y[idx];
    r.yaw = d.track_yaw[idx];
    // Rect::length is the extent along the yaw (along-track) and Rect::width the
    // extent across it. The crate is 0.25 m along-track by 0.35 m across-track.
    r.length = kCrateWidthM;
    r.width = kCrateLengthM;
    return r;
}

double waiting_offset(const TrackHandler& track, double s, int side) {
    const double width = side > 0 ? track.width_left(s) : track.width_right(s);
    return side * (width - 0.5 * kCrateWidthM - kWaitGapM);
}

Rect moving_obstacle(const TrackHandler& track, double s, double n, double vx, double vy) {
    Rect r = obstacle_at(track, s, n);
    r.vx = vx;
    r.vy = vy;
    return r;
}

Request straight_request(const State& z) {
    Request rq;
    rq.z = z;
    rq.u_rl = Input{0.0, z.v};
    rq.u_prev = rq.u_rl;
    rq.u_before = rq.u_rl;
    complete_state(&rq.z, rq.u_prev);
    return rq;
}

} // namespace

int main() {
    TrackHandler track;
    if (!track.load(kTrackPath)) {
        std::printf("cannot load %s (run with working directory sf/)\n", kTrackPath);
        return 1;
    }
    std::printf("== predictive filter ==\n");
    const double s0 = 3.6;

    {
        Filter filter(&track, kVMaxMps);
        const Request rq = straight_request(pose_at(track, station_at_s(track, s0), 0.0, 3.0));
        const Result r = filter.step(rq);
        check(r.outcome == Outcome::Passthrough, "open track: RL command passes through");
        check(!r.engaged && r.side == PassSide::None, "open track: no obstacle engaged");
        check(r.minimum_clearance_rl > 0.0, "open track: positive certificate margin");
    }

    {
        Filter filter(&track, kVMaxMps);
        Request rq = straight_request(pose_at(track, station_at_s(track, s0), 0.0, 3.0));
        rq.obstacles.push_back(obstacle_at(track, s0 + 2.5, 0.15));
        const Result r = filter.step(rq);
        check(r.engaged, "near obstacle: engaged");
        check(r.outcome != Outcome::Uncertified, "near obstacle, RL straight: a family certifies");
        check(r.side == PassSide::Right, "near obstacle: passes on the wide (right) side");
        check(r.u.delta_t <= 0.0, "near obstacle: published steering does not turn left");
        check(r.minimum_clearance_published >= 0.0, "near obstacle: published rollout is clear");
    }

    {
        Filter filter(&track, kVMaxMps);
        Request rq = straight_request(pose_at(track, station_at_s(track, s0), 0.0, 3.0));
        rq.obstacles.push_back(obstacle_at(track, s0 + 3.5, 0.0));
        const Result r = filter.step(rq);
        check(r.outcome == Outcome::Passthrough, "far obstacle: RL still passes through");
    }

    {
        Filter filter(&track, kVMaxMps);
        Request rq = straight_request(pose_at(track, station_at_s(track, s0), 0.0, 3.0));
        rq.obstacles.push_back(obstacle_at(track, s0 + 1.0, -2.5));
        const Result r = filter.step(rq);
        check(!r.engaged && r.outcome == Outcome::Passthrough, "off-track rect ignored");
    }

    {
        Filter filter(&track, kVMaxMps);
        State z = pose_at(track, station_at_s(track, s0), 0.0, 3.0);
        Input prev{0.0, 3.0};
        Input before = prev;
        bool consistent = true;
        double offset_at_obstacle = 0.0;
        for (int k = 0; k < 20; ++k) {
            Request rq;
            rq.z = z;
            rq.u_rl = Input{0.0, 3.0};
            rq.u_prev = prev;
            rq.u_before = before;
            complete_state(&rq.z, prev);
            rq.obstacles.push_back(obstacle_at(track, s0 + 2.5, 0.15));
            const Result r = filter.step(rq);
            if (r.outcome != Outcome::Uncertified &&
                r.minimum_clearance_published < 0.0) {
                consistent = false;
            }
            z = step(z, before, prev, r.u, kControlPeriodS, kVMaxMps);
            before = prev;
            prev = r.u;
            const Frenet f = track.to_frenet(z.x, z.y, z.psi);
            if (std::abs(f.s - (s0 + 2.5)) < 0.4) {
                offset_at_obstacle = std::max(offset_at_obstacle, std::abs(f.n));
            }
        }
        check(consistent, "certified outcomes carry a non-negative margin");
        check(offset_at_obstacle > 0.30, "closed loop passed beside the obstacle");
    }

    {
        Filter filter(&track, kVMaxMps);
        Request rq = straight_request(pose_at(track, station_at_s(track, s0), 0.0, 3.0));
        const TrackData& d = track.data();
        const double s_c = s0 + 2.5;
        const int idx = station_at_s(track, s_c);
        rq.obstacles.push_back(moving_obstacle(track,
                                               s_c,
                                               waiting_offset(track, s_c, 1),
                                               -kCrateSpeedMps * d.n_x[idx],
                                               -kCrateSpeedMps * d.n_y[idx]));
        const Result r = filter.step(rq);
        check(r.engaged, "crossing obstacle: engaged");
        check(r.outcome != Outcome::Uncertified, "crossing obstacle: a family certifies");
        check(r.minimum_clearance_published >= 0.0, "crossing obstacle: published rollout is clear");
    }

    {
        Filter filter(&track, kVMaxMps);
        Request rq = straight_request(pose_at(track, station_at_s(track, s0), 0.0, 3.0));
        const TrackData& d = track.data();
        const double s_c = s0 + 4.5;
        const int idx = station_at_s(track, s_c);
        rq.obstacles.push_back(moving_obstacle(track,
                                               s_c,
                                               waiting_offset(track, s_c, -1),
                                               -kCrateSpeedMps * std::cos(d.track_yaw[idx]),
                                               -kCrateSpeedMps * std::sin(d.track_yaw[idx])));
        const Result r = filter.step(rq);
        check(r.engaged, "oncoming obstacle: engaged");
        check(r.outcome != Outcome::Uncertified, "oncoming obstacle: a family certifies");
        check(r.minimum_clearance_published >= 0.0, "oncoming obstacle: published rollout is clear");
    }

    {
        Filter filter(&track, kVMaxMps);
        Request rq = straight_request(pose_at(track, station_at_s(track, s0), 0.0, 3.0));
        const double s_c = s0 + 2.5;
        rq.obstacles.push_back(obstacle_at(track, s_c, waiting_offset(track, s_c, -1)));
        const Result r = filter.step(rq);
        check(r.outcome == Outcome::Passthrough, "crossed obstacle: RL command passes through");
        check(r.minimum_clearance_published >= 0.0, "crossed obstacle: published rollout is clear");
    }

    {
        Filter filter_all(&track, kVMaxMps);
        Filter filter_ahead(&track, kVMaxMps);
        Filter filter_just_passed(&track, kVMaxMps);
        const State z = pose_at(track, station_at_s(track, s0), 0.0, 3.0);
        const Rect behind = obstacle_at(track, s0 - 0.4, waiting_offset(track, s0 - 0.4, -1));
        const Rect nearest = obstacle_at(track, s0 + 1.2, waiting_offset(track, s0 + 1.2, -1));
        const Rect farther = obstacle_at(track, s0 + 2.5, 0.0);
        Request rq_all = straight_request(z);
        rq_all.obstacles = {behind, nearest, farther};
        Request rq_ahead = straight_request(z);
        rq_ahead.obstacles = {nearest, farther};
        Request rq_just_passed = straight_request(z);
        rq_just_passed.obstacles = {behind, nearest};
        const Result r_all = filter_all.step(rq_all);
        const Result r_ahead = filter_ahead.step(rq_ahead);
        const Result r_just_passed = filter_just_passed.step(rq_just_passed);
        check(near(r_all.constraint_values_published[kConstraintObstacle],
                   r_ahead.constraint_values_published[kConstraintObstacle], 1e-12) &&
                  near(r_all.u.delta_t, r_ahead.u.delta_t, 1e-12) &&
                  r_all.engaged == r_ahead.engaged,
              "three crates: the two nearest ahead are admitted");
        check(!near(r_all.constraint_values_published[kConstraintObstacle],
                    r_just_passed.constraint_values_published[kConstraintObstacle], 1e-3) ||
                  r_all.engaged != r_just_passed.engaged,
              "three crates: the just-passed crate is not one of the two admitted");
    }

    {
        State z{};
        z.v = 3.0;
        const Input prev{0.0, 3.0};
        const Input before{0.2, 3.0};
        complete_state(&z, prev);
        const State two_command = step(z, prev, prev, steer_switch_s(), kVMaxMps);
        const State three_command = predict_backup_start(z, before, prev);
        const double delta_diff = std::abs(three_command.delta - two_command.delta);
        const double carry_s = steer_switch_s() - kControlPeriodS;
        State via_carry = step(z, before, before, before, carry_s, kVMaxMps);
        via_carry = step(via_carry, prev, prev, prev, kControlPeriodS, kVMaxMps);
        check(delta_diff > 0.015 && delta_diff < 0.050,
              "backup start delta differs from the two-command form by ~0.035 rad");
        check(near(three_command.delta, via_carry.delta, 1e-9) &&
                  near(three_command.delta_dot, via_carry.delta_dot, 1e-9),
              "backup start matches integrate_segment with the carry");
    }

    {
        Filter filter(&track, kVMaxMps);
        Request rq = straight_request(pose_at(track, station_at_s(track, s0), 0.0, 3.0));
        rq.obstacles.push_back(obstacle_at(track, s0 + 2.5, 0.15));
        const Result r = filter.step(rq);
        const State& z_term = r.rollout.back();
        const Frenet f = track.to_frenet(z_term.x, z_term.y, z_term.psi);
        // The row reads the body heading alone; the carried sideslip is left out.
        const double theta = wrap_pi(f.alpha);
        const double half =
            0.5 * (kEgoLengthM * std::abs(std::sin(f.alpha)) +
                   kEgoWidthM * std::abs(std::cos(f.alpha)));
        const double centre = std::min(track.width_left(f.s) - f.n,
                                       track.width_right(f.s) + f.n);
        const double room = std::max(0.0, centre - half);
        const double v_term = std::max(z_term.v, kKinematicFloorMps);
        const double a_term = SafetyParameters::lat_budget_mps2 -
                              std::abs(track.kappa_s(f.s)) * v_term * v_term;
        const double expect =
            lateral_stop_heading_limit(a_term, room, v_term) - std::abs(theta);
        check(near(r.constraint_values_published[kConstraintLateralStop], expect, 1e-9),
              "published lateral-stop row matches the heading-dependent room");
    }

    std::printf("%s (%d failing checks)\n", failures ? "FAILURES" : "OK", failures);
    return failures ? 1 : 0;
}
