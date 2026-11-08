#pragma once

// Calibrated single-track vehicle and actuator prediction model.

namespace sf {

struct State {
    double x = 0.0;
    double y = 0.0;
    double psi = 0.0;
    double delta = 0.0; // axle steering angle                  [rad]
    double v = 0.0;
    double r = 0.0;    // yaw rate                             [rad/s]
    double beta = 0.0; // body sideslip                        [rad]
    double fy_f = 0.0; // relaxed axle lateral forces          [N]
    double fy_r = 0.0;
    double delta_dot = 0.0; // steering servo rate state            [rad/s]
    double accel = 0.0;     // acceleration lag state               [m/s^2]
};

struct Input {
    double delta_t = 0.0; // axle steering target (after the gamma curve) [rad]
    double v_des = 0.0;   // speed setpoint                              [m/s]
};

// Steady-state tyre solution at (delta, v).
struct SteadyStateCornering {
    double a_y = 0.0;  // lateral acceleration                     [m/s^2]
    double beta = 0.0; // body sideslip                            [rad]
    double r = 0.0;    // yaw rate, a_y / v                        [rad/s]
    bool saturated = false;
};

double wrap_pi(double angle_rad);

// Return the tyre response corresponding to lateral acceleration.
double tyre_response_from_lateral_accel(double a_y_mps2);
// Steering angle at which steady-state cornering reaches the tyre limit.
double delta_sat_rad(double v_mps);
// Steering equation, forward: the axle angle whose steady-state lateral accel is a_y.
double delta_for_ay(double a_y_mps2, double v_mps);
// Steering equation inverted, with the saturation branch.
SteadyStateCornering steady_state_cornering(double delta_rad, double v_mps);
// Largest axle angle whose steady-state a_y stays within `a_lat`.
double delta_max_rad(double v_mps, double a_lat_mps2);

// Return rear lateral capacity remaining under longitudinal acceleration.
double rear_lateral_accel_limit(double a_x_mps2);

// Gamma curve: steering command <-> axle target.
double axle_target_from_command(double delta_cmd_rad, double delta_th_rad, double steer_gamma);
double command_from_axle_target(double delta_t_rad, double delta_th_rad, double steer_gamma);
void assert_gamma_monotone(double delta_th_rad, double steer_gamma);

// Estimate unmeasured model state from measurements and the active command.
void complete_state(State* z, const Input& active);

// Advance one control period while delayed actuators switch commands.
// `before`, `prev`, `next` are the commands published two periods ago, one
// period ago, and at the start of this period; a switch time longer than the
// period is honoured exactly. `v_max_mps` is the speed the drivetrain stops
// accelerating at.
State step(const State& z, const Input& before, const Input& prev,
           const Input& next, double dt_s, double v_max_mps);
// Same with the command before `prev` taken equal to `prev`.
State step(const State& z, const Input& prev, const Input& next, double dt_s,
           double v_max_mps);

// Delay from a command's publish time to each actuator: compute latency plus
// transport. It may exceed one control period.
double steer_switch_s();
double speed_switch_s();

} // namespace sf
