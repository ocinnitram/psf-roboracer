#pragma once

#include "sf_model.h"
#include "sf_rect.h"
#include "sf_tuning.h"
#include "track.h"

#include <vector>

// Keep the RL command when its predicted path is safe; otherwise search for a
// safer steering and speed command, falling back to the backup command.

namespace sf {

constexpr double kInf = 1e30;

struct Request {
  State z; // measured state completed by complete_state()
  Input u_rl;
  Input u_prev;   // command published one period ago, still in the pipeline
  Input u_before; // command published two periods ago: with a steering delay
                  // longer than the period it is the one active at period start
  std::vector<Rect> obstacles; // rect.vx/vy hold tracked velocity [m/s]
  // Also record Result::commands (telemetry only; decisions are unchanged).
  bool export_commands = false;
};

// Safety constraints of one rollout, each in its own unit.
enum SafetyConstraint {
  kConstraintObstacle,
  kConstraintWall,
  kConstraintSpeed,
  kConstraintLateralStop,
  kConstraintCone,
  kConstraintProgress,
  kConstraintCount
};

// How the published command was selected against the backup command.
enum class Outcome {
  Passthrough, // RL command certified
  Qp,          // QP solution certified
  Segment,     // point between the QP solution and backup certified
  Backup,      // the backup command itself published
  Uncertified  // backup violates a row; least-violating command published
};

enum class PassSide : int {
  Right = -1,
  None = 0,
  Left = 1
};

struct Result {
  Input u;
  Outcome outcome = Outcome::Uncertified;
  double constraint_values_rl[kConstraintCount];
  double constraint_values_published[kConstraintCount];
  double minimum_clearance_rl = -kInf;
  double minimum_clearance_published = -kInf;
  bool engaged = false;       // an obstacle steers the backup
  PassSide side = PassSide::None;
  bool stop = false;          // selected backup option brakes to rest
  int stages = 0;             // rollout length of the published command
  int rollouts = 0;           // rollouts evaluated this period
  const char *reason = "ok";  // most violated row of the published rollout
  std::vector<State> rollout;
  // Command entering each stage of `rollout` (commands[j-1] drives stage j:
  // u for stage 1, the backup command after). Only with
  // Request::export_commands; size rollout.size() - 1.
  std::vector<Input> commands;
};

class Filter {
public:
  // `v_max_mps` is the top speed the certificate is built for: it caps the
  // corridor speed and the QP's speed row, and sets the rollout horizon.
  Filter(const TrackHandler *track, double v_max_mps);

  Result step(const Request &request);
  void reset();

  double v_max_mps() const { return m_limits.v_max; }
  int min_stages() const { return m_limits.min_stages; }
  int max_stages() const { return m_limits.max_stages; }

private:
  struct TrackPose;
  struct Obstacle;
  struct PredictedObstacle;
  struct Rollout;
  struct EvaluatedCommand;
  struct BackupEvaluation;
  struct CommandSearchResult;
  // One pass-side or stop backup option.
  struct BackupOption {
    PassSide side = PassSide::None;
    bool stop = false;
  };
  // Remembered pass side for the nearest engaged obstacle.
  struct RememberedPassSide {
    double x = 0.0;
    double y = 0.0;
    PassSide side = PassSide::None;
  };

  TrackPose track_pose(const State &z) const;
  double ahead_m(double s_from, double s_to) const;
  // Admit obstacles and express their motion and pass room in track
  // coordinates.
  bool obstacle_in_track_frame(const Rect &rect, const TrackPose &now,
                               Obstacle *obstacle) const;
  void measure_pass_room(Obstacle *obstacle) const;
  PassSide choose_side(const Obstacle &obstacle, bool holds_latch) const;
  std::vector<Obstacle> admit_obstacles(const Request &request,
                                        const TrackPose &now);
  // Backup policy.
  double reference_offset(const TrackPose &car, double car_v,
                          const std::vector<Obstacle> &obstacles,
                          const BackupOption &option, double elapsed_s) const;
  double brake_reach_m(double v) const;
  double corridor_speed(double s, double v) const;
  double brake_room_mps2(const State &z, double a_y_mps2) const;
  double speed_floor_mps(const State &z) const;
  Input backup_command(const State &z, const Input &before, const Input &active,
                       const std::vector<Obstacle> &obstacles,
                       const BackupOption &option, double now_s) const;
  void update_stage_constraints(
      const State &z, const State &prev, const TrackPose &pose,
      const std::vector<Obstacle> &obstacles, double elapsed_s,
      double values[kConstraintCount]) const;
  void evaluate_terminal_constraints(
      const State &z_end, const TrackPose &end_pose,
      const std::vector<Obstacle> &obstacles, double elapsed_s,
      double values[kConstraintCount]) const;
  Rollout predict_constraints(const Request &request, const Input &command,
                              const std::vector<Obstacle> &obstacles,
                              const BackupOption &option) const;
  EvaluatedCommand evaluate_command(const Request &request,
                                    const std::vector<Obstacle> &obstacles,
                                    const BackupOption &option, const Input &u,
                                    int *rollouts) const;
  BackupEvaluation evaluate_backup_option(
      const Request &request, const std::vector<Obstacle> &obstacles,
      const BackupOption &option, int *rollouts) const;
  bool qp_step(const Request &request, const std::vector<Obstacle> &obstacles,
               const BackupEvaluation &evaluation,
               const EvaluatedCommand &current, int *rollouts,
               EvaluatedCommand *next) const;
  EvaluatedCommand bisect_toward_backup(
      const Request &request, const std::vector<Obstacle> &obstacles,
      const BackupEvaluation &evaluation, const Input &from,
      int *rollouts) const;
  EvaluatedCommand correct_for_backup_option(
      const Request &request, const std::vector<Obstacle> &obstacles,
      const BackupEvaluation &evaluation, int *rollouts) const;
  CommandSearchResult select_best_command(
      const Request &request, const std::vector<Obstacle> &obstacles,
      const Obstacle *nearest, int *rollouts) const;

  const TrackHandler *m_track;
  SpeedLimits m_limits;
  RememberedPassSide m_remembered_pass_side;
  // Carry unmeasured model state from the previous published rollout.
  State m_predicted;
  bool m_have_predicted = false;
};

} // namespace sf
