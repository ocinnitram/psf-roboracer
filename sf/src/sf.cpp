// Safety-filter facade: gamma-curve I/O, state completion, command history,
// and the output the eval records. The certificate itself is sf_filter.cpp.

#include "sf.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vehicle/params.h"
#include "sf_geometry.h"
#include "sf_math.h"
#include "sf_model.h"
#include "sf_tuning.h"

namespace sf {

namespace {

// Rectangle -> two circles along its long axis; radius covers the quarter
// rectangle so the pair never under-covers. `along_m` is the extent along yaw.
Circles rect_to_circles(double cx, double cy, double along_m, double across_m, double yaw) {
    const double lo = std::min(along_m, across_m);
    const double hi = std::max(along_m, across_m);
    const double axis = (along_m >= across_m) ? yaw : yaw + kHalfPi;
    const double off = 0.25 * hi;
    const double c = std::cos(axis);
    const double s = std::sin(axis);
    return {cx + off * c, cy + off * s, cx - off * c, cy - off * s, std::hypot(off, 0.5 * lo)};
}

constexpr double kEmptyObstacleCircleCoordM = 50.0;
constexpr double kEmptyObstacleCircleRadiusM = 0.1;
constexpr int kExportedObstacleSlots = 2;

// Describe why the filter published its selected command.
CommandSelection command_selection(const Result& result) {
    if (result.outcome == Outcome::Passthrough) {
        return result.engaged ? CommandSelection::CertifiedRl
                              : CommandSelection::NoObstacleInHorizon;
    }
    if (result.outcome == Outcome::Uncertified) {
        return CommandSelection::Uncertified;
    }
    if (result.stop) {
        return CommandSelection::Stop;
    }
    if (result.side == PassSide::Left) {
        return CommandSelection::PassLeft;
    }
    if (result.side == PassSide::Right) {
        return CommandSelection::PassRight;
    }
    return CommandSelection::Corridor;
}

// Reject a design speed the certificate cannot be built for before
// SpeedLimits derives the stop horizon and the corridor cap from it.
double validated_v_max_mps(double v_max_mps) {
    if (!std::isfinite(v_max_mps) || v_max_mps <= 0.0) {
        throw std::invalid_argument("SafetyFilter v_max_mps must be finite and positive, got " +
                                    std::to_string(v_max_mps));
    }
    return v_max_mps;
}

HiddenState hidden_state(const State& z) {
    return {z.beta, z.fy_f, z.fy_r, z.delta_dot, z.accel};
}

} // namespace

SafetyFilter::SafetyFilter(const TrackHandler& track, double v_max_mps)
    : m_filter(&track, validated_v_max_mps(v_max_mps)) {
    assert_gamma_monotone(kSteerDeltaThRad, kSteerGamma);
}

SafetyFilterOutput SafetyFilter::certify(const VehicleState& measured,
                                         const std::vector<Rect>& obstacles,
                                         const Action& desired,
                                         bool include_diagnostics) {
    Request request;
    request.z.x = measured.x;
    request.z.y = measured.y;
    request.z.psi = measured.yaw;
    request.z.delta = measured.delta;
    request.z.v = measured.v;
    request.z.r = measured.r;
    request.u_rl.delta_t =
        axle_target_from_command(desired.delta, kSteerDeltaThRad, kSteerGamma);
    // The design speed is a hard limit: a request above it is never certified
    // as-is, whatever the prediction says (the model saturates there).
    const double v_max = m_filter.v_max_mps();
    const bool rl_speed_capped = desired.v > v_max;
    request.u_rl.v_des = rl_speed_capped ? v_max : desired.v;
    request.u_prev = m_has_published ? m_published : request.u_rl;
    request.u_before = m_has_published ? m_published_before : request.u_rl;
    complete_state(&request.z, request.u_prev);
    const HiddenState steady = hidden_state(request.z);
    request.obstacles = obstacles;
    request.export_commands = include_diagnostics;

    const Result result = m_filter.step(request);
    m_published_before = m_has_published ? m_published : result.u;
    m_published = result.u;
    m_has_published = true;

    SafetyFilterOutput out;
    out.command.delta =
        command_from_axle_target(result.u.delta_t, kSteerDeltaThRad, kSteerGamma);
    out.command.v = result.u.v_des;
    out.outcome = result.outcome;
    out.rl_speed_capped = rl_speed_capped;

    SafetyFilterDiagnostics diagnostics;
    diagnostics.rollouts = result.rollouts;
    diagnostics.failed_constraint = result.reason;
    // Filter::step starts every rollout from the carried-over hidden states.
    diagnostics.hidden_estimate =
        result.rollout.empty() ? steady : hidden_state(result.rollout.front());
    diagnostics.hidden_steady_state = steady;
    for (int i = 0; i < kConstraintCount; ++i) {
        diagnostics.constraint_values_rl[static_cast<std::size_t>(i)] =
            result.constraint_values_rl[i];
        diagnostics.constraint_values_published[static_cast<std::size_t>(i)] =
            result.constraint_values_published[i];
    }
    diagnostics.selection = command_selection(result);
    if (!include_diagnostics) {
        out.diagnostics = std::move(diagnostics);
        return out;
    }

    const std::size_t exported =
        std::min(result.rollout.size(),
                 static_cast<std::size_t>(m_filter.max_stages() + 1));
    for (std::size_t i = 0; i < exported; ++i) {
        const State& z = result.rollout[i];
        VehicleState st;
        st.x = z.x;
        st.y = z.y;
        st.yaw = z.psi;
        st.delta = z.delta;
        st.v = z.v;
        st.r = z.r;
        diagnostics.predicted_states.push_back(st);
    }
    const std::size_t exported_commands =
        std::min(result.commands.size(), exported > 0 ? exported - 1 : 0);
    diagnostics.predicted_commands.reserve(exported_commands);
    for (std::size_t i = 0; i < exported_commands; ++i) {
        const Input& u = result.commands[i];
        diagnostics.predicted_commands.push_back(
            {command_from_axle_target(u.delta_t, kSteerDeltaThRad, kSteerGamma),
             u.v_des});
    }

    for (int i = 0; i < kExportedObstacleSlots; ++i) {
        if (i < static_cast<int>(obstacles.size())) {
            const Rect& r = obstacles[i];
            diagnostics.obstacle_circles.push_back(
                rect_to_circles(r.cx, r.cy, r.length, r.width, r.yaw));
        } else {
            diagnostics.obstacle_circles.push_back(
                {kEmptyObstacleCircleCoordM, kEmptyObstacleCircleCoordM,
                 kEmptyObstacleCircleCoordM, kEmptyObstacleCircleCoordM,
                 kEmptyObstacleCircleRadiusM});
        }
    }
    const double ego_circle_radius = cover_disk_radius();
    const Rotation heading = rotation(request.z.psi);
    for (int i = 0; i < SafetyParameters::cover_disks; ++i) {
        const Point centre = cover_disk_centre(request.z, heading, i);
        diagnostics.ego_circles.push_back(
            {centre.x, centre.y, ego_circle_radius});
    }
    out.diagnostics = std::move(diagnostics);
    return out;
}

void SafetyFilter::set_published(const Action& action) {
    Input u;
    u.delta_t =
        axle_target_from_command(action.delta, kSteerDeltaThRad, kSteerGamma);
    u.v_des = action.v;
    m_published_before = m_has_published ? m_published : u;
    m_published = u;
    m_has_published = true;
}

void SafetyFilter::reset() {
    m_has_published = false;
    m_filter.reset();
}

} // namespace sf
