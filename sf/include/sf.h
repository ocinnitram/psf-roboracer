#pragma once

#include "sf_filter.h"
#include "sf_rect.h"
#include "track.h"

#include <array>
#include <optional>
#include <string>
#include <vector>

// Public safety-filter facade: gamma-curve I/O, command history, and the
// telemetry a renderer or eval records. The certificate itself is Filter.

namespace sf {

struct VehicleState {
    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
    double delta = 0.0; // axle steering angle [rad]
    double v = 0.0;
    double r = 0.0;
};

// Steering and speed expressed in the policy/drive-command space.
struct Action {
    double delta = 0.0;
    double v = 0.0;
};

// One certification or renderer circle.
struct Circle {
    double x = 0.0;
    double y = 0.0;
    double r = 0.0;
};

// Two-circle renderer approximation of an obstacle rectangle.
struct Circles {
    double x1 = 0.0;
    double y1 = 0.0;
    double x2 = 0.0;
    double y2 = 0.0;
    double r = 0.0;
};

enum class CommandSelection {
    NoObstacleInHorizon,
    CertifiedRl,
    Stop,
    PassLeft,
    PassRight,
    Corridor,
    Uncertified
};

// The states the vehicle does not measure, as the filter holds them.
struct HiddenState {
    double beta = 0.0;      // body slip angle [rad]
    double fy_f = 0.0;      // relaxed front lateral force [N]
    double fy_r = 0.0;      // relaxed rear lateral force [N]
    double delta_dot = 0.0; // steering rate [rad/s]
    double accel = 0.0;     // longitudinal acceleration [m/s^2]
};

// Per-period telemetry. Always present: rollouts, failed_constraint,
// hidden states, barrier rows, selection. Only with include_diagnostics: the
// rollout (states and commands) and the renderer circles.
struct SafetyFilterDiagnostics {
    // Rollouts (simulations) evaluated this period.
    int rollouts = 0;
    // Most violated row of the published rollout (kConstraintName), "ok" when
    // every row holds.
    std::string failed_constraint = "ok";
    std::vector<VehicleState> predicted_states;
    // Command (policy/drive-command space) entering each stage of the
    // published rollout: [0] = published command, [j>0] = the backup command
    // the rollout applied. Size predicted_states.size() - 1.
    std::vector<Action> predicted_commands;
    // Hidden states the rollouts started from: carried over from the previous
    // tick's first predicted period (steady state on the first tick).
    HiddenState hidden_estimate;
    // The steady-state completion of the same measured state, for comparison.
    HiddenState hidden_steady_state;
    std::vector<Circles> obstacle_circles;
    std::vector<Circle> ego_circles;
    std::array<double, kConstraintCount> constraint_values_rl{};
    std::array<double, kConstraintCount> constraint_values_published{};
    CommandSelection selection = CommandSelection::Uncertified;
};

// Control result; certify always sets diagnostics (see above for which
// fields need include_diagnostics).
struct SafetyFilterOutput {
    Action command;
    Outcome outcome = Outcome::Uncertified;
    // The policy's speed was above v_max_mps and was certified as v_max_mps:
    // outcome, selection and constraint_values_rl describe the capped command.
    bool rl_speed_capped = false;
    std::optional<SafetyFilterDiagnostics> diagnostics;
};

class SafetyFilter {
public:
    // `v_max_mps` is the top speed the certificate is built for; it caps the
    // policy's speed and sets the corridor cap, the QP's speed row, and the
    // rollout horizon. Borrows `track`. Throws std::invalid_argument unless
    // `v_max_mps` is finite and positive.
    SafetyFilter(const TrackHandler& track, double v_max_mps);

    SafetyFilter(const SafetyFilter&) = delete;
    SafetyFilter& operator=(const SafetyFilter&) = delete;

    double v_max_mps() const { return m_filter.v_max_mps(); }
    int min_stages() const { return m_filter.min_stages(); }
    int max_stages() const { return m_filter.max_stages(); }

    SafetyFilterOutput certify(const VehicleState& state,
                               const std::vector<Rect>& obstacles,
                               const Action& u_rl,
                               bool include_diagnostics = false);

    // Override the stored published command so the next certify uses it as
    // u_prev (and the previous stored command as u_before). Action is in
    // policy/drive-command space, converted with the gamma curve.
    void set_published(const Action& action);

    void reset();

private:
    Filter m_filter;
    Input m_published;
    Input m_published_before;
    bool m_has_published = false;
};

} // namespace sf
