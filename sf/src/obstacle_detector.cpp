// LiDAR obstacle detector: wall mask, clustering, L-shape fit with a square
// prior, admission of on-track boxes.

#include "obstacle_detector.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef SF_DEBUG_UDP
#include <arpa/inet.h>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "nanoflann.hpp"

namespace sf {

struct ObstacleDetector::TrackAdmissionIndex {
  TrackData data;
  struct Adaptor {
    const TrackData &track;
    explicit Adaptor(const TrackData &t) : track(t) {}
    // Report the number of centreline samples to nanoflann.
    std::size_t kdtree_get_point_count() const { return track.x.size(); }
    // Return one coordinate of a centreline sample to nanoflann.
    double kdtree_get_pt(const std::size_t idx, const std::size_t dim) const {
      if (dim == 0) {
        return track.x[idx];
      }
      return track.y[idx];
    }
    // nanoflann adaptor API requires this member template.
    template <class Bbox> bool kdtree_get_bbox(Bbox &) const { return false; }
  };
  using Tree = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, Adaptor>, Adaptor, 2>;
  Adaptor adaptor;
  Tree index;

  // Copy validated track arrays and build their nearest-point index.
  explicit TrackAdmissionIndex(const TrackData &track_data)
      : data(track_data), adaptor(data),
        index(2, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10)) {
    if (data.x.empty() || data.x.size() != data.y.size() ||
        data.x.size() != data.n_x.size() || data.x.size() != data.n_y.size() ||
        data.x.size() != data.w_l.size() || data.x.size() != data.w_r.size() ||
        data.x.size() != data.s.size() ||
        data.x.size() != data.track_yaw.size()) {
      throw std::invalid_argument("ObstacleDetector track data is incomplete");
    }
    index.buildIndex();
  }

  // Return the centreline sample nearest a world point.
  std::size_t nearest(double x, double y) const {
    const double query[2] = {x, y};
    std::size_t result_index = 0;
    double distance_squared = 0.0;
    nanoflann::KNNResultSet<double> result(1);
    result.init(&result_index, &distance_squared);
    index.findNeighbors(result, query, nanoflann::SearchParameters());
    return result_index;
  }
};

namespace {

#ifdef SF_DEBUG_UDP
constexpr int kDebugPacketPort = 9876;
constexpr const char *kDebugPacketHost = "127.0.0.1";
#endif

constexpr int kSearchAngleStartDeg = 0;
constexpr int kSearchAngleEndDeg = 90;
constexpr int kSearchAngleStepDeg = 2;

// Three-sigma range-noise allowance for adaptive scan clustering.
constexpr double kStochasticErrorSigma = 3.0;

// Mid-grey cutoff between dark rendered walls and light free space.
constexpr int kPgmOccupiedGreyCeiling = 128;

// Keeps points exactly at the clearance distance inside the margin.
constexpr double kWallClearanceEpsilonSq = 1e-12;

// No-candidate-yet sentinel; variances are non-negative so their negated
// sum can never beat it.
constexpr double kLShapeScoreSentinel = -1e12;

// A fit with no extent along either axis (every return at one point) is
// degenerate. A face seen without range noise is a line of zero depth; the
// square prior completes it.
constexpr double kMinRectExtentM = 1e-3;

// Median-gap floor so a split never divides by a vanishing median.
constexpr double kMinMedianGapSq = 1e-6;

// Build the default two-degree L-shape yaw search grid.
std::vector<int> default_search_angles() {
  std::vector<int> angles;
  for (int deg = kSearchAngleStartDeg; deg < kSearchAngleEndDeg;
       deg += kSearchAngleStepDeg) {
    angles.push_back(deg);
  }
  return angles;
}

// Read one non-comment token from a PGM stream.
std::string next_pgm_token(std::istream &input) {
  std::string token;
  while (input >> token) {
    if (!token.empty() && token.front() == '#') {
      input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }
    return token;
  }
  return {};
}

// Decode binary or ASCII PGM pixels into one byte per map cell.
bool read_pgm_pixels(std::istream &input, const std::string &magic, int width,
                     int height, std::vector<std::uint8_t> &pixels) {
  pixels.resize(static_cast<std::size_t>(width * height));
  if (magic == "P5") {
    input.get();
    input.read(reinterpret_cast<char *>(pixels.data()),
               static_cast<std::streamsize>(pixels.size()));
    return input.gcount() == static_cast<std::streamsize>(pixels.size());
  }
  for (std::uint8_t &pixel : pixels) {
    const std::string value = next_pgm_token(input);
    if (value.empty()) {
      return false;
    }
    pixel = static_cast<std::uint8_t>(std::stoi(value));
  }
  return true;
}

// Dilate occupied map cells into a precomputed wall-clearance mask.
void stamp_wall_clearance(std::vector<std::uint8_t> &wall_margin,
                          const std::vector<std::uint8_t> &occupied, int width,
                          int height, double resolution,
                          double wall_clearance) {
  const int radius = static_cast<int>(std::ceil(wall_clearance / resolution));
  const double clearance_squared =
      wall_clearance * wall_clearance + kWallClearanceEpsilonSq;
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      if (!occupied[static_cast<std::size_t>(row * width + col)]) {
        continue;
      }
      for (int dr = -radius; dr <= radius; ++dr) {
        for (int dc = -radius; dc <= radius; ++dc) {
          if ((dr * dr + dc * dc) * resolution * resolution >
              clearance_squared) {
            continue;
          }
          const int rr = row + dr;
          const int cc = col + dc;
          if (rr < 0 || rr >= height || cc < 0 || cc >= width) {
            continue;
          }
          wall_margin[static_cast<std::size_t>(rr * width + cc)] = 1;
        }
      }
    }
  }
}

#ifdef SF_DEBUG_UDP
// Send one debug payload to UDP port 9876 for an external plotter (not part of
// this repository).
void send_udp_debug_payload(const std::string &payload) {
  const int sock = socket(AF_INET, SOCK_DGRAM, 0);
  if (sock < 0) {
    return;
  }
  sockaddr_in addr;
  addr.sin_family = AF_INET;
  addr.sin_port = htons(kDebugPacketPort);
  addr.sin_addr.s_addr = inet_addr(kDebugPacketHost);
  sendto(sock, payload.c_str(), payload.size(), 0,
         reinterpret_cast<const sockaddr *>(&addr), sizeof(addr));
  close(sock);
}

// Serialize a pose and point cloud into a debug packet for the external
// plotter on UDP port 9876.
void send_debug_packet(const Pose2D &pose, const std::vector<Point2D> &points,
                       const std::string &color) {
  std::stringstream ss;
  ss << pose.x << "," << pose.y << "," << pose.yaw << "|" << color << "|";
  for (const Point2D &p : points) {
    ss << p.x << "," << p.y << ";";
  }
  send_udp_debug_payload(ss.str());
}
#endif

struct RotatedClusterFrame {
  std::vector<double> x;
  std::vector<double> y;
  double min_x = 0.0;
  double max_x = 0.0;
  double min_y = 0.0;
  double max_y = 0.0;
};

// Rotate one cluster into a candidate box frame and measure its bounds.
RotatedClusterFrame
rotate_cluster_to_search_frame(const std::vector<Point2D> &world_points,
                               double cos_a, double sin_a, int start_ndx,
                               int end_ndx) {
  RotatedClusterFrame frame;
  frame.min_x = std::numeric_limits<double>::infinity();
  frame.max_x = -std::numeric_limits<double>::infinity();
  frame.min_y = std::numeric_limits<double>::infinity();
  frame.max_y = -std::numeric_limits<double>::infinity();

  for (int ndx = start_ndx; ndx <= end_ndx; ndx++) {
    const double x_rot =
        world_points[ndx].x * cos_a + world_points[ndx].y * sin_a;
    const double y_rot =
        -world_points[ndx].x * sin_a + world_points[ndx].y * cos_a;
    frame.x.push_back(x_rot);
    frame.y.push_back(y_rot);
    if (x_rot < frame.min_x) {
      frame.min_x = x_rot;
    }
    if (x_rot > frame.max_x) {
      frame.max_x = x_rot;
    }
    if (y_rot < frame.min_y) {
      frame.min_y = y_rot;
    }
    if (y_rot > frame.max_y) {
      frame.max_y = y_rot;
    }
  }
  return frame;
}

// Assign each rotated point's distance to its nearest candidate box edge.
void partition_edge_distances(const RotatedClusterFrame &frame,
                              std::vector<double> &e1_dist,
                              std::vector<double> &e2_dist) {
  e1_dist.reserve(frame.x.size());
  e2_dist.reserve(frame.y.size());
  for (std::size_t j = 0; j < frame.x.size(); j++) {
    const double rx = frame.x[j];
    const double ry = frame.y[j];
    const double d1 =
        std::min(std::abs(rx - frame.min_x), std::abs(rx - frame.max_x));
    const double d2 =
        std::min(std::abs(ry - frame.min_y), std::abs(ry - frame.max_y));
    if (d1 < d2) {
      e1_dist.push_back(d1);
    } else {
      e2_dist.push_back(d2);
    }
  }
}

} // namespace

// Populate detector defaults that require a generated angle sequence.
DetectorConfig::DetectorConfig() : search_angles(default_search_angles()) {}

// Precompute scan geometry and allocate storage for repeated detections.
ObstacleDetector::ObstacleDetector(DetectorConfig cfg,
                                   const TrackHandler &track,
                                   BoundaryIndex boundary)
    : m_cfg(std::move(cfg)), m_boundary(std::move(boundary)) {
  if (m_cfg.search_angles.empty()) {
    m_cfg.search_angles = default_search_angles();
  }

  m_scan_angles.resize(static_cast<std::size_t>(m_cfg.n_beams));
  const double angle_increment = m_cfg.lidar_fov / (m_cfg.n_beams - 1);
  for (int i = 0; i < m_cfg.n_beams; i++) {
    m_scan_angles[static_cast<std::size_t>(i)] =
        -m_cfg.lidar_fov / 2.0 + i * angle_increment;
  }

  m_obstacle_ranges.reserve(static_cast<std::size_t>(m_cfg.n_beams));
  m_obstacle_angles.reserve(static_cast<std::size_t>(m_cfg.n_beams));
  m_obstacle_local_points.reserve(static_cast<std::size_t>(m_cfg.n_beams));
  m_obstacle_world_points.reserve(static_cast<std::size_t>(m_cfg.n_beams));
  m_diff_scan_dist_sq.reserve(static_cast<std::size_t>(m_cfg.n_beams));
  m_scan_clusters.reserve(static_cast<std::size_t>(m_cfg.n_beams));
  m_final_clusters.reserve(static_cast<std::size_t>(m_cfg.n_beams));
  m_cluster_gaps.reserve(static_cast<std::size_t>(m_cfg.n_beams));
  if (m_cfg.expected_obstacle_count > 0) {
    m_detected.items.reserve(
        static_cast<std::size_t>(m_cfg.expected_obstacle_count));
  }
  m_preprocessed_scans.resize(static_cast<std::size_t>(m_cfg.n_beams));
  m_world_points.resize(static_cast<std::size_t>(m_cfg.n_beams));

  for (int angle_deg : m_cfg.search_angles) {
    const double angle_rad = angle_deg * kPi / 180.0;
    m_cos_search_angles.push_back(std::cos(angle_rad));
    m_sin_search_angles.push_back(std::sin(angle_rad));
  }

  m_track_admission = std::make_unique<TrackAdmissionIndex>(track.data());
}

// Transfer ownership of a detector and its indexes.
ObstacleDetector::ObstacleDetector(ObstacleDetector &&) noexcept = default;
// Replace this detector with another detector's owned state.
ObstacleDetector &
ObstacleDetector::operator=(ObstacleDetector &&) noexcept = default;
// Release the detector's owned indexes and buffers.
ObstacleDetector::~ObstacleDetector() = default;

// Run the scan-to-box detection pipeline for one LiDAR frame.
RawDetections ObstacleDetector::detect(const std::vector<float> &ranges,
                                       const Pose2D &lidar_pose) {
  if (static_cast<int>(ranges.size()) != m_cfg.n_beams) {
    RawDetections invalid;
    invalid.pose = lidar_pose;
    invalid.valid = false;
    return invalid;
  }

  preprocess_scans(ranges, lidar_pose);
  cluster_adjacent_returns();
  split_merged_clusters();
  fit_shape(lidar_pose);
  m_detected.pose = lidar_pose;
  m_detected.valid = true;
  return m_detected;
}

// Load an occupancy raster and precompute its wall-clearance mask.
bool ObstacleDetector::set_static_map(const std::string &pgm_path,
                                      double resolution, double origin_x,
                                      double origin_y, double wall_clearance) {
  std::ifstream input(pgm_path, std::ios::binary);
  if (!input || resolution <= 0.0 || wall_clearance < 0.0) {
    return false;
  }

  const std::string magic = next_pgm_token(input);
  if (magic != "P5" && magic != "P2") {
    return false;
  }
  const std::string width_token = next_pgm_token(input);
  const std::string height_token = next_pgm_token(input);
  const std::string max_token = next_pgm_token(input);
  if (width_token.empty() || height_token.empty() || max_token.empty()) {
    return false;
  }
  const int width = std::stoi(width_token);
  const int height = std::stoi(height_token);
  const int maximum = std::stoi(max_token);
  if (width <= 0 || height <= 0 || maximum <= 0 || maximum > 255) {
    return false;
  }

  std::vector<std::uint8_t> pixels;
  if (!read_pgm_pixels(input, magic, width, height, pixels)) {
    return false;
  }

  m_static_map_width = width;
  m_static_map_height = height;
  m_static_map_resolution = resolution;
  m_static_map_origin_x = origin_x;
  m_static_map_origin_y = origin_y;
  m_static_map_occupied.assign(pixels.size(), 0);
  m_static_map_wall_margin.assign(pixels.size(), 0);
  for (std::size_t i = 0; i < pixels.size(); ++i) {
    if (pixels[i] <= kPgmOccupiedGreyCeiling) {
      m_static_map_occupied[i] = 1;
    }
  }

  stamp_wall_clearance(m_static_map_wall_margin, m_static_map_occupied, width,
                       height, resolution, wall_clearance);
  return true;
}

// Convert a world point to an in-bounds occupancy-map cell.
bool ObstacleDetector::map_point_to_pixel(double x, double y, int &col,
                                          int &row) const {
  if (m_static_map_width <= 0 || m_static_map_height <= 0) {
    return false;
  }
  col = static_cast<int>(
      std::floor((x - m_static_map_origin_x) / m_static_map_resolution));
  const int map_y = static_cast<int>(
      std::floor((y - m_static_map_origin_y) / m_static_map_resolution));
  row = m_static_map_height - 1 - map_y;
  return col >= 0 && col < m_static_map_width && row >= 0 &&
         row < m_static_map_height;
}

// Test whether a fitted rectangle lies within the local track widths.
bool ObstacleDetector::rectangle_inside_track(const Rect &rect) const {
  if (!m_track_admission) {
    return true;
  }
  const std::size_t i = m_track_admission->nearest(rect.cx, rect.cy);
  const TrackData &track = m_track_admission->data;
  const double centre_lateral = (rect.cx - track.x[i]) * track.n_x[i] +
                                (rect.cy - track.y[i]) * track.n_y[i];
  const double yaw_x = std::cos(rect.yaw);
  const double yaw_y = std::sin(rect.yaw);
  // length along yaw, width perpendicular — project the box onto the
  // local track normal.
  const double lateral_radius =
      0.5 *
      (rect.length * std::abs(yaw_x * track.n_x[i] + yaw_y * track.n_y[i]) +
       rect.width * std::abs(-yaw_y * track.n_x[i] + yaw_x * track.n_y[i]));
  return centre_lateral - lateral_radius >= -track.w_r[i] &&
         centre_lateral + lateral_radius <= track.w_l[i];
}

// Trace LiDAR-to-box line of sight through the occupancy raster.
bool ObstacleDetector::static_map_line_is_clear(const Pose2D &lidar_pose,
                                                const Rect &rect) const {
  if (m_static_map_occupied.empty()) {
    return true;
  }
  int x0, y0, x1, y1;
  if (!map_point_to_pixel(lidar_pose.x, lidar_pose.y, x0, y0) ||
      !map_point_to_pixel(rect.cx, rect.cy, x1, y1)) {
    return false;
  }

  const int target_x = x1;
  const int target_y = y1;
  const int dx = std::abs(x1 - x0);
  const int sx = x0 < x1 ? 1 : -1;
  const int dy = -std::abs(y1 - y0);
  const int sy = y0 < y1 ? 1 : -1;
  int error = dx + dy;
  while (x0 != target_x || y0 != target_y) {
    const int twice_error = 2 * error;
    if (twice_error >= dy) {
      error += dy;
      x0 += sx;
    }
    if (twice_error <= dx) {
      error += dx;
      y0 += sy;
    }
    if (x0 == target_x && y0 == target_y) {
      break;
    }
    if (m_static_map_occupied[static_cast<std::size_t>(y0 * m_static_map_width +
                                                       x0)]) {
      return false;
    }
  }
  return true;
}

// Test whether a box centre lies outside the precomputed wall margin.
bool ObstacleDetector::static_map_wall_clearance_is_clear(
    const Rect &rect) const {
  if (m_static_map_wall_margin.empty()) {
    return true;
  }
  int col, row;
  if (!map_point_to_pixel(rect.cx, rect.cy, col, row)) {
    return false;
  }
  return !m_static_map_wall_margin[static_cast<std::size_t>(
      row * m_static_map_width + col)];
}

// Apply the pose-independent track and wall-margin admission checks.
bool ObstacleDetector::accepts(const Rect &rect) const {
  return rectangle_inside_track(rect) &&
         static_map_wall_clearance_is_clear(rect);
}

// Compute a sample median without fully sorting the values.
double ObstacleDetector::sample_median(std::vector<double> sample) {
  if (sample.empty()) {
    return 0.0;
  }
  const std::size_t n = sample.size();
  const std::size_t mid = n / 2;
  std::nth_element(sample.begin(),
                   sample.begin() + static_cast<std::ptrdiff_t>(mid),
                   sample.end());
  if (n % 2 != 0) {
    return sample[mid];
  }
  const double mid1 = sample[mid];
  std::nth_element(sample.begin(),
                   sample.begin() + static_cast<std::ptrdiff_t>(mid - 1),
                   sample.end());
  const double mid2 = sample[mid - 1];
  return (mid1 + mid2) / 2.0;
}

// Compute the population variance of a sample.
double ObstacleDetector::sample_variance(const std::vector<double> &data) {
  if (data.size() < 2) {
    return 0.0;
  }
  double mean = 0.0;
  for (double value : data) {
    mean += value;
  }
  mean /= static_cast<double>(data.size());
  double variance = 0.0;
  for (double value : data) {
    variance += (value - mean) * (value - mean);
  }
  variance /= static_cast<double>(data.size());
  return variance;
}

// Transform valid non-wall returns into local and world obstacle points.
void ObstacleDetector::preprocess_scans(const std::vector<float> &ranges,
                                        const Pose2D &lidar_pose) {
  m_obstacle_angles.clear();
  m_obstacle_ranges.clear();
  m_obstacle_local_points.clear();
  m_obstacle_world_points.clear();

  const double cos_yaw = std::cos(lidar_pose.yaw);
  const double sin_yaw = std::sin(lidar_pose.yaw);

  for (int i = 0; i < static_cast<int>(ranges.size()); i++) {
    const std::size_t idx = static_cast<std::size_t>(i);
    const double range = static_cast<double>(ranges[idx]);
    // No return: inf from the sensor, or max_range after the node's clean-up.
    if (range >= m_cfg.max_range) {
      m_preprocessed_scans[idx] = std::numeric_limits<double>::infinity();
      m_world_points[idx].x = std::numeric_limits<double>::infinity();
      m_world_points[idx].y = std::numeric_limits<double>::infinity();
      continue;
    }

    m_world_points[idx].x =
        lidar_pose.x + range * std::cos(m_scan_angles[idx] + lidar_pose.yaw);
    m_world_points[idx].y =
        lidar_pose.y + range * std::sin(m_scan_angles[idx] + lidar_pose.yaw);

    const double dx = m_world_points[idx].x - lidar_pose.x;
    const double dy = m_world_points[idx].y - lidar_pose.y;
    Point2D local_points;
    local_points.x = dx * cos_yaw + dy * sin_yaw;
    local_points.y = -dx * sin_yaw + dy * cos_yaw;

    if (local_points.x <= 0.0) {
      m_preprocessed_scans[idx] = std::numeric_limits<double>::infinity();
      m_world_points[idx].x = std::numeric_limits<double>::infinity();
      m_world_points[idx].y = std::numeric_limits<double>::infinity();
      continue;
    }

    const double dist_to_wall =
        m_boundary.distance(m_world_points[idx].x, m_world_points[idx].y);
    const double wall_mask_distance =
        m_cfg.eps_wall + range * std::abs(std::sin(m_cfg.wall_yaw_error));
    if (dist_to_wall < wall_mask_distance) {
      m_preprocessed_scans[idx] = std::numeric_limits<double>::infinity();
      continue;
    }

    m_preprocessed_scans[idx] = range;
    m_obstacle_local_points.push_back(local_points);
    m_obstacle_world_points.push_back(m_world_points[idx]);
    m_obstacle_angles.push_back(m_scan_angles[idx]);
    m_obstacle_ranges.push_back(range);
  }
#ifdef SF_DEBUG_UDP
  if (m_cfg.debug) {
    send_debug_packet(lidar_pose, m_obstacle_world_points, "red");
  }
#endif
}

// Compute the adaptive breakpoint distance between adjacent scan returns.
double ObstacleDetector::adaptive_gap_limit_sq(int i) const {
  const std::size_t idx = static_cast<std::size_t>(i);
  const double d_phi = m_obstacle_angles[idx] - m_obstacle_angles[idx - 1];
  if ((m_cfg.breakpoint_angle_rad - d_phi) <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  const double div_const =
      std::sin(d_phi) / std::sin(m_cfg.breakpoint_angle_rad - d_phi);
  return std::pow(m_obstacle_ranges[idx - 1] * div_const +
                      kStochasticErrorSigma * m_cfg.lidar_stochastic_error,
                  2);
}

// Test whether a new scan cluster is close enough to merge backward.
bool ObstacleDetector::is_mergeable_with_previous(int start_index) const {
  if (m_scan_clusters.empty()) {
    return false;
  }
  const int previous_end = m_scan_clusters.back().cluster_end;
  const double dx_clusters =
      m_obstacle_local_points[static_cast<std::size_t>(start_index)].x -
      m_obstacle_local_points[static_cast<std::size_t>(previous_end)].x;
  const double dy_clusters =
      m_obstacle_local_points[static_cast<std::size_t>(start_index)].y -
      m_obstacle_local_points[static_cast<std::size_t>(previous_end)].y;
  const double merge_dist_sq =
      dx_clusters * dx_clusters + dy_clusters * dy_clusters;
  return merge_dist_sq < m_cfg.cluster_merge_threshold;
}

// Group adjacent scan returns and merge nearby neighbouring clusters.
void ObstacleDetector::cluster_adjacent_returns() {
  m_diff_scan_dist_sq.clear();
  m_scan_clusters.clear();
  ScanCluster save_cluster;

  if (static_cast<int>(m_obstacle_ranges.size()) < m_cfg.min_cluster_size) {
    return;
  }

  int current_cluster_start = 0;
  for (int i = 1; i < static_cast<int>(m_obstacle_angles.size()); i++) {
    const double breakpoint_sq = adaptive_gap_limit_sq(i);
    const double dx =
        m_obstacle_local_points[static_cast<std::size_t>(i)].x -
        m_obstacle_local_points[static_cast<std::size_t>(i - 1)].x;
    const double dy =
        m_obstacle_local_points[static_cast<std::size_t>(i)].y -
        m_obstacle_local_points[static_cast<std::size_t>(i - 1)].y;
    m_diff_scan_dist_sq.push_back(dx * dx + dy * dy);

    if (m_diff_scan_dist_sq[static_cast<std::size_t>(i - 1)] <= breakpoint_sq) {
      continue;
    }
    if ((i - current_cluster_start) < m_cfg.min_cluster_size) {
      current_cluster_start = i;
      continue;
    }
    if (!m_scan_clusters.empty() &&
        is_mergeable_with_previous(current_cluster_start)) {
      m_scan_clusters.back().cluster_end = i - 1;
      current_cluster_start = i;
      continue;
    }
    save_cluster.cluster_start = current_cluster_start;
    save_cluster.cluster_end = i - 1;
    m_scan_clusters.push_back(save_cluster);
    current_cluster_start = i;
  }

  if ((static_cast<int>(m_obstacle_angles.size()) - 1 - current_cluster_start) <
      m_cfg.min_cluster_size) {
    return;
  }
  if (!m_scan_clusters.empty() &&
      is_mergeable_with_previous(current_cluster_start)) {
    m_scan_clusters.back().cluster_end =
        static_cast<int>(m_obstacle_angles.size()) - 1;
    return;
  }
  save_cluster.cluster_start = current_cluster_start;
  save_cluster.cluster_end = static_cast<int>(m_obstacle_angles.size()) - 1;
  m_scan_clusters.push_back(save_cluster);
}

// Split merged clusters at an outlying internal point gap.
void ObstacleDetector::split_merged_clusters() {
  m_final_clusters.clear();
  for (const ScanCluster &cluster : m_scan_clusters) {
    const int start_ndx = cluster.cluster_start;
    const int end_ndx = cluster.cluster_end;
    m_cluster_gaps.clear();
    double max_gap_sq = -1.0;
    int max_gap_ndx = -1;
    for (int k = start_ndx; k < end_ndx; k++) {
      const double gap = m_diff_scan_dist_sq[static_cast<std::size_t>(k)];
      m_cluster_gaps.push_back(gap);
      if (gap > max_gap_sq) {
        max_gap_sq = gap;
        max_gap_ndx = k;
      }
    }

    const double median_gap_sq = sample_median(m_cluster_gaps);
    if (median_gap_sq > kMinMedianGapSq &&
        max_gap_sq > (m_cfg.split_factor * median_gap_sq) &&
        max_gap_sq > m_cfg.min_split_dist_sq) {
      ScanCluster cluster1;
      cluster1.cluster_start = start_ndx;
      cluster1.cluster_end = max_gap_ndx;
      ScanCluster cluster2;
      cluster2.cluster_start = max_gap_ndx + 1;
      cluster2.cluster_end = end_ndx;
      if ((cluster1.cluster_end - cluster1.cluster_start + 1) >=
          m_cfg.min_cluster_size) {
        m_final_clusters.push_back(cluster1);
      }
      if ((cluster2.cluster_end - cluster2.cluster_start + 1) >=
          m_cfg.min_cluster_size) {
        m_final_clusters.push_back(cluster2);
      }
    } else {
      m_final_clusters.push_back(cluster);
    }
  }
}

// Search candidate yaws for the minimum-variance L-shape rectangle.
void ObstacleDetector::fit_l_shape(int start_index, int end_index,
                                   Rect &fitted_rect, double &best_cos,
                                   double &best_sin) {
  double best_score = kLShapeScoreSentinel;
  for (std::size_t i = 0; i < m_cos_search_angles.size(); i++) {
    const RotatedClusterFrame frame = rotate_cluster_to_search_frame(
        m_obstacle_world_points, m_cos_search_angles[i], m_sin_search_angles[i],
        start_index, end_index);
    std::vector<double> e1_dist;
    std::vector<double> e2_dist;
    partition_edge_distances(frame, e1_dist, e2_dist);
    const double score = -(sample_variance(e1_dist)) - sample_variance(e2_dist);
    if (score > best_score) {
      best_score = score;
      // Search-frame x is the candidate yaw; length lies along yaw.
      fitted_rect.length = frame.max_x - frame.min_x;
      fitted_rect.width = frame.max_y - frame.min_y;
      const double centre_rot_x = (frame.min_x + frame.max_x) / 2.0;
      const double centre_rot_y = (frame.min_y + frame.max_y) / 2.0;
      fitted_rect.cx = centre_rot_x * m_cos_search_angles[i] -
                       centre_rot_y * m_sin_search_angles[i];
      fitted_rect.cy = centre_rot_x * m_sin_search_angles[i] +
                       centre_rot_y * m_cos_search_angles[i];
      fitted_rect.yaw = m_cfg.search_angles[i] * kPi / 180.0;
      best_cos = m_cos_search_angles[i];
      best_sin = m_sin_search_angles[i];
    }
  }
}

// Complete a one-face fit with a square prior while preserving its visible edge.
void ObstacleDetector::apply_square_prior(RawDetection &detection,
                                          Rect &l_shape, double best_cos,
                                          double best_sin,
                                          const Pose2D &lidar_pose) {
  const double max_dim = std::max(l_shape.length, l_shape.width);
  const double min_dim = std::min(l_shape.length, l_shape.width);
  if ((min_dim / max_dim) >= m_cfg.aspect_ratio_threshold) {
    return;
  }

  detection.single_side_estimate = true;
  const double dx = lidar_pose.x - l_shape.cx;
  const double dy = lidar_pose.y - l_shape.cy;
  const double ego_rot_x = dx * best_cos + dy * best_sin;
  const double ego_rot_y = -dx * best_sin + dy * best_cos;
  const double shift_dist = (max_dim - min_dim) / 2.0;
  double shift_rot_x = 0.0;
  double shift_rot_y = 0.0;

  if (l_shape.width < l_shape.length) {
    detection.length_is_prior = false;
    if (ego_rot_y < 0.0) {
      shift_rot_y = shift_dist;
    } else {
      shift_rot_y = -shift_dist;
    }
    l_shape.width = max_dim;
  } else {
    detection.length_is_prior = true;
    if (ego_rot_x < 0.0) {
      shift_rot_x = shift_dist;
    } else {
      shift_rot_x = -shift_dist;
    }
    l_shape.length = max_dim;
  }

  l_shape.cx += shift_rot_x * best_cos - shift_rot_y * best_sin;
  l_shape.cy += shift_rot_x * best_sin + shift_rot_y * best_cos;
}

// Tangent yaw of the nearest forward centreline point, as a prior for the
// tracker's yaw. The fitted box is left as measured; NaN without a track.
double ObstacleDetector::track_yaw_prior(const Rect &rect,
                                         const Pose2D &lidar_pose) const {
  if (!m_track_admission) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const TrackData &track = m_track_admission->data;
  const std::size_t ego_index =
      m_track_admission->nearest(lidar_pose.x, lidar_pose.y);
  const double ego_s = track.s[ego_index];
  const double track_length = track.s.back();
  std::size_t track_index = ego_index;
  double best_distance_sq = std::numeric_limits<double>::infinity();
  for (std::size_t j = 0; j < track.s.size(); ++j) {
    double forward_s = track.s[j] - ego_s;
    if (forward_s < 0.0) {
      forward_s += track_length;
    }
    if (forward_s > m_cfg.max_range) {
      continue;
    }
    const double dx = track.x[j] - rect.cx;
    const double dy = track.y[j] - rect.cy;
    const double distance_sq = dx * dx + dy * dy;
    if (distance_sq < best_distance_sq) {
      best_distance_sq = distance_sq;
      track_index = j;
    }
  }
  return track.track_yaw[track_index];
}

// Fit and admit one rectangle for each final scan cluster.
void ObstacleDetector::fit_shape(const Pose2D &lidar_pose) {
  m_detected.items.clear();
  for (const ScanCluster &cluster : m_final_clusters) {
    Rect l_shape{};
    double best_cos = 0.0;
    double best_sin = 0.0;
    fit_l_shape(cluster.cluster_start, cluster.cluster_end, l_shape, best_cos,
                best_sin);
    if (std::max(l_shape.length, l_shape.width) < kMinRectExtentM) {
      continue;
    }
    RawDetection detection;
    detection.support_points = cluster.cluster_end - cluster.cluster_start + 1;
    apply_square_prior(detection, l_shape, best_cos, best_sin, lidar_pose);
    detection.rect = l_shape;
    detection.yaw_prior = track_yaw_prior(l_shape, lidar_pose);
    if (!accepts(detection.rect) ||
        !static_map_line_is_clear(lidar_pose, detection.rect)) {
      continue;
    }
    m_detected.items.push_back(detection);
  }
}

} // namespace sf
