#pragma once

#include "boundary.h"
#include "vehicle/params.h"
#include "sf_math.h"
#include "sf_rect.h"
#include "track.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace sf {

struct Pose2D {
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
};

// LiDAR clustering and L-shape fitting knobs. Defaults are the values eval
// writes onto the C++ config on a nominal (not sim2real) run.
struct DetectorConfig {
  // Send debug packets to UDP port 9876; compiled only with SF_DEBUG_UDP. [-]
  bool debug = false;
  // Expected fitted boxes, used only to preallocate output storage.    [-]
  int expected_obstacle_count = 2;
  // Scan span used to place beam angles.                               [rad]
  double lidar_fov = kLidarFovRad;
  // Native beam count the scan vector must match.                      [-]
  int n_beams = kLidarNumBeams;
  // Range noise used by adaptive scan clustering.                     [m]
  double lidar_stochastic_error = 0.02;
  // Range-independent wall-mask slack (map tolerance, no sim2real).    [m]
  double eps_wall = 0.08;
  // Extra wall-mask slack from heading noise; 0 on a nominal run.      [rad]
  double wall_yaw_error = 0.0;
  // Hits beyond this are discarded before clustering.                  [m]
  double max_range = 10.0;
  // Maximum angle used to separate adjacent scan returns.              [rad]
  double breakpoint_angle_rad = 170.0 * kPi / 180.0;
  // Squared-distance gate for merging adjacent scan clusters.          [m^2]
  double cluster_merge_threshold = 0.75;
  // Minimum hits in a kept cluster.                                    [-]
  int min_cluster_size = 3;
  // Split a cluster when its max gap exceeds this times the median.    [-]
  double split_factor = 10.0;
  // Absolute squared-gap floor for a split.                            [m^2]
  double min_split_dist_sq = 0.75;
  // Below this min/max extent ratio, apply the square prior.           [-]
  double aspect_ratio_threshold = 0.35;
  // L-shape search yaw samples in degrees; default 0,2,...,88.
  std::vector<int> search_angles;

  DetectorConfig();
};

// One fitted box plus the measurement metadata the tracker needs.
struct RawDetection {
  Rect rect;
  int support_points = 0;            // LiDAR hits in the fitted scan cluster
  bool single_side_estimate = false; // square prior applied (one face visible)
  bool length_is_prior = false;      // length guessed; only when single-side
  // Track tangent near the box: the tracker's initial yaw estimate, never a
  // replacement for rect.yaw. NaN when the detector has no track.
  double yaw_prior = std::numeric_limits<double>::quiet_NaN();
};

// Detector output for one scan. valid is false on beam-count mismatch; the
// tracker then treats the scan as empty.
struct RawDetections {
  std::vector<RawDetection> items;
  Pose2D pose;
  bool valid = true;
};

class ObstacleDetector {
public:
  // Build track admission data and take ownership of the boundary index.
  ObstacleDetector(DetectorConfig cfg, const TrackHandler &track,
                   BoundaryIndex boundary);
  ObstacleDetector(const ObstacleDetector &) = delete;
  ObstacleDetector &operator=(const ObstacleDetector &) = delete;
  ObstacleDetector(ObstacleDetector &&) noexcept;
  ObstacleDetector &operator=(ObstacleDetector &&) noexcept;
  ~ObstacleDetector();

  // Cluster, fit, and admit obstacles from one scan. ranges.size() must
  // equal cfg.n_beams; otherwise returns valid=false and does not throw.
  RawDetections detect(const std::vector<float> &ranges,
                       const Pose2D &lidar_pose);

  // Load a PGM occupancy raster for line-of-sight and wall-margin checks.
  // wall_clearance is metres in map coordinates. False on a bad file.
  bool set_static_map(const std::string &pgm_path, double resolution,
                      double origin_x, double origin_y, double wall_clearance);

  // Pose-independent admission: on-track and off the occupancy wall margin.
  // Line-of-sight from the LiDAR is applied inside detect, which has a pose.
  bool accepts(const Rect &rect) const;

private:
  struct TrackAdmissionIndex;
  struct ScanCluster {
    int cluster_start = 0;
    int cluster_end = 0;
  };

  DetectorConfig m_cfg;
  BoundaryIndex m_boundary;
  std::vector<double> m_scan_angles;

  std::vector<double> m_obstacle_ranges;
  std::vector<double> m_obstacle_angles;
  std::vector<Point2D> m_obstacle_local_points;
  std::vector<double> m_preprocessed_scans;
  std::vector<Point2D> m_world_points;
  std::vector<Point2D> m_obstacle_world_points;
  std::vector<double> m_diff_scan_dist_sq;
  std::vector<ScanCluster> m_scan_clusters;
  std::vector<ScanCluster> m_final_clusters;
  std::vector<double> m_cluster_gaps;
  std::vector<double> m_cos_search_angles;
  std::vector<double> m_sin_search_angles;
  RawDetections m_detected;

  std::unique_ptr<TrackAdmissionIndex> m_track_admission;
  std::vector<std::uint8_t> m_static_map_occupied;
  std::vector<std::uint8_t> m_static_map_wall_margin;
  int m_static_map_width = 0;
  int m_static_map_height = 0;
  double m_static_map_resolution = 0.0;
  double m_static_map_origin_x = 0.0;
  double m_static_map_origin_y = 0.0;

  void preprocess_scans(const std::vector<float> &ranges,
                        const Pose2D &lidar_pose);
  void cluster_adjacent_returns();
  void split_merged_clusters();
  void fit_shape(const Pose2D &lidar_pose);
  void fit_l_shape(int start_index, int end_index, Rect &fitted_rect,
                   double &best_cos, double &best_sin);
  void apply_square_prior(RawDetection &detection, Rect &l_shape,
                          double best_cos, double best_sin,
                          const Pose2D &lidar_pose);
  double track_yaw_prior(const Rect &rect, const Pose2D &lidar_pose) const;
  double adaptive_gap_limit_sq(int i) const;
  bool is_mergeable_with_previous(int start_index) const;
  double sample_median(std::vector<double> sample);
  double sample_variance(const std::vector<double> &data);
  bool rectangle_inside_track(const Rect &rect) const;
  bool map_point_to_pixel(double x, double y, int &col, int &row) const;
  bool static_map_line_is_clear(const Pose2D &lidar_pose,
                                const Rect &rect) const;
  bool static_map_wall_clearance_is_clear(const Rect &rect) const;
};

} // namespace sf
