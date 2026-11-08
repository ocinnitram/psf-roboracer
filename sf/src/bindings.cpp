// Python module ppo_sf_roboracer: track, detector, tracker, and safety filter.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "boundary.h"
#include "obstacle_detector.h"
#include "obstacle_tracker.h"
#include "sf.h"
#include "sf_model.h"
#include "sf_rect.h"
#include "track.h"
#include "vehicle/params.h"

#include <utility>

namespace py = pybind11;

PYBIND11_MODULE(ppo_sf_roboracer, m) {
  m.doc() = "Safety filter, track, LiDAR detector, and obstacle tracker.";

  // track.h
  py::class_<sf::Point2D>(m, "Point2D")
      .def(py::init<>())
      .def_readwrite("x", &sf::Point2D::x)
      .def_readwrite("y", &sf::Point2D::y);

  py::class_<sf::Frenet>(m, "Frenet")
      .def(py::init<>())
      .def_readwrite("s", &sf::Frenet::s)
      .def_readwrite("n", &sf::Frenet::n)
      .def_readwrite("alpha", &sf::Frenet::alpha);

  py::class_<sf::TrackData>(m, "TrackData")
      .def_property_readonly("x", [](const sf::TrackData &t) { return t.x; })
      .def_property_readonly("y", [](const sf::TrackData &t) { return t.y; })
      .def_property_readonly("s", [](const sf::TrackData &t) { return t.s; })
      .def_property_readonly("s_norm",
                             [](const sf::TrackData &t) { return t.s_norm; })
      .def_property_readonly("track_yaw",
                             [](const sf::TrackData &t) { return t.track_yaw; })
      .def_property_readonly("kappa",
                             [](const sf::TrackData &t) { return t.kappa; })
      .def_property_readonly("kappa_s",
                             [](const sf::TrackData &t) { return t.kappa_s; })
      .def_property_readonly("w_l",
                             [](const sf::TrackData &t) { return t.w_l; })
      .def_property_readonly("w_r",
                             [](const sf::TrackData &t) { return t.w_r; })
      .def_property_readonly("n_x",
                             [](const sf::TrackData &t) { return t.n_x; })
      .def_property_readonly("n_y",
                             [](const sf::TrackData &t) { return t.n_y; });

  py::class_<sf::TrackHandler>(m, "TrackHandler")
      .def(py::init<>())
      .def("load", &sf::TrackHandler::load, py::arg("csv_path"))
      .def_property_readonly("data", &sf::TrackHandler::data,
                             py::return_value_policy::reference_internal)
      .def("to_frenet", &sf::TrackHandler::to_frenet, py::arg("x"),
           py::arg("y"), py::arg("yaw"));

  // sf_rect.h
  py::class_<sf::Rect>(m, "Rect")
      .def(py::init<>())
      .def_readwrite("cx", &sf::Rect::cx)
      .def_readwrite("cy", &sf::Rect::cy)
      .def_readwrite("yaw", &sf::Rect::yaw)
      .def_readwrite("length", &sf::Rect::length)
      .def_readwrite("width", &sf::Rect::width)
      .def_readwrite("vx", &sf::Rect::vx)
      .def_readwrite("vy", &sf::Rect::vy);

  // sf.h
  py::class_<sf::VehicleState>(m, "VehicleState")
      .def(py::init<>())
      .def_readwrite("x", &sf::VehicleState::x)
      .def_readwrite("y", &sf::VehicleState::y)
      .def_readwrite("yaw", &sf::VehicleState::yaw)
      .def_readwrite("delta", &sf::VehicleState::delta)
      .def_readwrite("v", &sf::VehicleState::v)
      .def_readwrite("r", &sf::VehicleState::r);

  py::class_<sf::Action>(m, "Action")
      .def(py::init<>())
      .def_readwrite("delta", &sf::Action::delta)
      .def_readwrite("v", &sf::Action::v);

  py::enum_<sf::Outcome>(m, "Outcome")
      .value("Passthrough", sf::Outcome::Passthrough)
      .value("Qp", sf::Outcome::Qp)
      .value("Segment", sf::Outcome::Segment)
      .value("Backup", sf::Outcome::Backup)
      .value("Uncertified", sf::Outcome::Uncertified);

  py::enum_<sf::PassSide>(m, "PassSide")
      .value("None", sf::PassSide::None)
      .value("Left", sf::PassSide::Left)
      .value("Right", sf::PassSide::Right);

  py::class_<sf::Circles>(m, "Circles")
      .def(py::init<>())
      .def_readwrite("x1", &sf::Circles::x1)
      .def_readwrite("y1", &sf::Circles::y1)
      .def_readwrite("x2", &sf::Circles::x2)
      .def_readwrite("y2", &sf::Circles::y2)
      .def_readwrite("r", &sf::Circles::r);

  py::class_<sf::Circle>(m, "Circle")
      .def(py::init<>())
      .def_readwrite("x", &sf::Circle::x)
      .def_readwrite("y", &sf::Circle::y)
      .def_readwrite("r", &sf::Circle::r);

  py::enum_<sf::CommandSelection>(m, "CommandSelection")
      .value("NoObstacleInHorizon",
             sf::CommandSelection::NoObstacleInHorizon)
      .value("CertifiedRl", sf::CommandSelection::CertifiedRl)
      .value("Stop", sf::CommandSelection::Stop)
      .value("PassLeft", sf::CommandSelection::PassLeft)
      .value("PassRight", sf::CommandSelection::PassRight)
      .value("Corridor", sf::CommandSelection::Corridor)
      .value("Uncertified", sf::CommandSelection::Uncertified);

  py::class_<sf::HiddenState>(m, "HiddenState")
      .def(py::init<>())
      .def_readwrite("beta", &sf::HiddenState::beta)
      .def_readwrite("fy_f", &sf::HiddenState::fy_f)
      .def_readwrite("fy_r", &sf::HiddenState::fy_r)
      .def_readwrite("delta_dot", &sf::HiddenState::delta_dot)
      .def_readwrite("accel", &sf::HiddenState::accel);

  py::class_<sf::SafetyFilterDiagnostics>(m, "SafetyFilterDiagnostics")
      .def(py::init<>())
      .def_readwrite("rollouts", &sf::SafetyFilterDiagnostics::rollouts)
      .def_readwrite("failed_constraint",
                     &sf::SafetyFilterDiagnostics::failed_constraint)
      .def_readwrite("predicted_states",
                     &sf::SafetyFilterDiagnostics::predicted_states)
      .def_readwrite("predicted_commands",
                     &sf::SafetyFilterDiagnostics::predicted_commands)
      .def_readwrite("hidden_estimate",
                     &sf::SafetyFilterDiagnostics::hidden_estimate)
      .def_readwrite("hidden_steady_state",
                     &sf::SafetyFilterDiagnostics::hidden_steady_state)
      .def_readwrite("obstacle_circles",
                     &sf::SafetyFilterDiagnostics::obstacle_circles)
      .def_readwrite("ego_circles",
                     &sf::SafetyFilterDiagnostics::ego_circles)
      .def_readwrite("constraint_values_rl",
                     &sf::SafetyFilterDiagnostics::constraint_values_rl)
      .def_readwrite("constraint_values_published",
                     &sf::SafetyFilterDiagnostics::constraint_values_published)
      .def_readwrite("selection", &sf::SafetyFilterDiagnostics::selection);

  py::class_<sf::SafetyFilterOutput>(m, "SafetyFilterOutput")
      .def(py::init<>())
      .def_readwrite("command", &sf::SafetyFilterOutput::command)
      .def_readwrite("outcome", &sf::SafetyFilterOutput::outcome)
      .def_readwrite("rl_speed_capped",
                     &sf::SafetyFilterOutput::rl_speed_capped)
      .def_readwrite("diagnostics", &sf::SafetyFilterOutput::diagnostics);

  py::class_<sf::SafetyFilter>(m, "SafetyFilter")
      .def(py::init<const sf::TrackHandler &, double>(), py::arg("track"),
           py::arg("v_max_mps"), py::keep_alive<1, 2>())
      .def_property_readonly("v_max_mps", &sf::SafetyFilter::v_max_mps)
      .def_property_readonly("min_stages", &sf::SafetyFilter::min_stages)
      .def_property_readonly("max_stages", &sf::SafetyFilter::max_stages)
      .def("certify", &sf::SafetyFilter::certify, py::arg("state"),
           py::arg("obstacles"), py::arg("u_rl"),
           py::arg("include_diagnostics") = false)
      .def("set_published", &sf::SafetyFilter::set_published, py::arg("action"))
      .def("reset", &sf::SafetyFilter::reset);

  m.def(
      "axle_from_command",
      [](double delta_cmd_rad) {
        return sf::axle_target_from_command(delta_cmd_rad, sf::kSteerDeltaThRad,
                                            sf::kSteerGamma);
      },
      py::arg("delta_cmd_rad"));

  // boundary.h
  py::class_<sf::BoundaryIndex>(m, "BoundaryIndex")
      .def(py::init<std::vector<sf::Point2D>>(), py::arg("points"))
      .def("distance", &sf::BoundaryIndex::distance, py::arg("x"),
           py::arg("y"));

  // obstacle_detector.h
  py::class_<sf::Pose2D>(m, "Pose2D")
      .def(py::init<>())
      .def_readwrite("x", &sf::Pose2D::x)
      .def_readwrite("y", &sf::Pose2D::y)
      .def_readwrite("yaw", &sf::Pose2D::yaw);

  py::class_<sf::DetectorConfig>(m, "DetectorConfig")
      .def(py::init<>())
      .def_readwrite("debug", &sf::DetectorConfig::debug)
      .def_readwrite("expected_obstacle_count",
                     &sf::DetectorConfig::expected_obstacle_count)
      .def_readwrite("lidar_fov", &sf::DetectorConfig::lidar_fov)
      .def_readwrite("n_beams", &sf::DetectorConfig::n_beams)
      .def_readwrite("lidar_stochastic_error",
                     &sf::DetectorConfig::lidar_stochastic_error)
      .def_readwrite("eps_wall", &sf::DetectorConfig::eps_wall)
      .def_readwrite("wall_yaw_error", &sf::DetectorConfig::wall_yaw_error)
      .def_readwrite("max_range", &sf::DetectorConfig::max_range)
      .def_readwrite("breakpoint_angle_rad",
                     &sf::DetectorConfig::breakpoint_angle_rad)
      .def_readwrite("cluster_merge_threshold",
                     &sf::DetectorConfig::cluster_merge_threshold)
      .def_readwrite("min_cluster_size", &sf::DetectorConfig::min_cluster_size)
      .def_readwrite("split_factor", &sf::DetectorConfig::split_factor)
      .def_readwrite("min_split_dist_sq",
                     &sf::DetectorConfig::min_split_dist_sq)
      .def_readwrite("aspect_ratio_threshold",
                     &sf::DetectorConfig::aspect_ratio_threshold)
      .def_readwrite("search_angles", &sf::DetectorConfig::search_angles);

  py::class_<sf::RawDetection>(m, "RawDetection")
      .def(py::init<>())
      .def_readwrite("rect", &sf::RawDetection::rect)
      .def_readwrite("support_points", &sf::RawDetection::support_points)
      .def_readwrite("single_side_estimate",
                     &sf::RawDetection::single_side_estimate)
      .def_readwrite("length_is_prior", &sf::RawDetection::length_is_prior)
      .def_readwrite("yaw_prior", &sf::RawDetection::yaw_prior);

  py::class_<sf::RawDetections>(m, "RawDetections")
      .def(py::init<>())
      .def_readwrite("items", &sf::RawDetections::items)
      .def_readwrite("pose", &sf::RawDetections::pose)
      .def_readwrite("valid", &sf::RawDetections::valid);

  py::class_<sf::ObstacleDetector>(m, "ObstacleDetector")
      .def(py::init([](sf::DetectorConfig config, const sf::TrackHandler &track,
                       sf::BoundaryIndex &boundary) {
             return new sf::ObstacleDetector(std::move(config), track,
                                             std::move(boundary));
           }),
           py::arg("config"), py::arg("track"), py::arg("boundary"),
           py::keep_alive<1, 3>())
      .def("detect", &sf::ObstacleDetector::detect, py::arg("ranges"),
           py::arg("lidar_pose"))
      .def("set_static_map", &sf::ObstacleDetector::set_static_map,
           py::arg("pgm_path"), py::arg("resolution"), py::arg("origin_x"),
           py::arg("origin_y"), py::arg("wall_clearance"))
      .def("accepts", &sf::ObstacleDetector::accepts, py::arg("rect"));

  // obstacle_tracker.h
  py::class_<sf::ObstacleTracker>(m, "ObstacleTracker")
      .def(py::init<>())
      .def("update", &sf::ObstacleTracker::update, py::arg("raw_detections"))
      .def("reset", &sf::ObstacleTracker::reset);
}
