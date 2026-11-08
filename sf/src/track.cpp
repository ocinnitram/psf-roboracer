// Load, smooth, interpolate, and project closed-loop track data.

#include "track.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "nanoflann.hpp"
#include "sf_math.h"

namespace sf {

namespace {

constexpr double kKappaSmoothWindowM = 0.10;
constexpr int kKdLeafSize = 10;

struct TrackPointAdaptor {
  const TrackData &obj;

  inline std::size_t kdtree_get_point_count() const { return obj.x.size(); }

  inline double kdtree_get_pt(const std::size_t idx,
                              const std::size_t dim) const {
    if (dim == 0) {
      return obj.x[idx];
    }
    return obj.y[idx];
  }

  template <class BBOX> bool kdtree_get_bbox(BBOX & /* bb */) const {
    return false;
  }
};

using KdTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, TrackPointAdaptor>, TrackPointAdaptor,
    2>;

std::vector<double> smooth_kappa_closed_loop(const std::vector<double> &kappa,
                                             double station_spacing_m) {
  const int half_width =
      std::max(0, static_cast<int>(std::lround((kKappaSmoothWindowM / 2.0) /
                                               station_spacing_m)));
  std::vector<double> smoothed(kappa.size(), 0.0);
  for (std::size_t i = 0; i < kappa.size(); ++i) {
    double sum = 0.0;
    for (int j = -half_width; j <= half_width; ++j) {
      const std::size_t idx = (static_cast<long long>(i) + j +
                               static_cast<long long>(kappa.size())) %
                              kappa.size();
      sum += kappa[idx];
    }
    smoothed[i] = sum / static_cast<double>(2 * half_width + 1);
  }
  return smoothed;
}

void clear_track_data(TrackData *data) {
  data->x.clear();
  data->y.clear();
  data->s.clear();
  data->s_norm.clear();
  data->track_yaw.clear();
  data->kappa.clear();
  data->kappa_s.clear();
  data->w_l.clear();
  data->w_r.clear();
  data->n_x.clear();
  data->n_y.clear();
}

} // namespace

struct TrackHandler::Index {
  TrackPointAdaptor adaptor;
  KdTree tree;

  explicit Index(const TrackData &data)
      : adaptor{data},
        tree(2, adaptor,
             nanoflann::KDTreeSingleIndexAdaptorParams(kKdLeafSize)) {
    tree.buildIndex();
  }
};

TrackHandler::TrackHandler() = default;

TrackHandler::~TrackHandler() = default;

bool TrackHandler::load(const std::string &csv_path) {
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    return false;
  }

  clear_track_data(&m_data);
  m_index.reset();
  m_length = 0.0;

  std::string line;
  std::getline(file, line);

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;
    std::vector<double> row;
    while (std::getline(ss, cell, ',')) {
      row.push_back(std::stod(cell));
    }
    if (row.size() >= 10) {
      m_data.x.push_back(row[0]);
      m_data.y.push_back(row[1]);
      m_data.s.push_back(row[2]);
      m_data.s_norm.push_back(row[3]);
      m_data.track_yaw.push_back(row[4]);
      m_data.kappa.push_back(row[5]);
      m_data.w_l.push_back(row[6]);
      m_data.w_r.push_back(row[7]);
      m_data.n_x.push_back(row[8]);
      m_data.n_y.push_back(row[9]);
    }
  }

  if (m_data.x.empty()) {
    return false;
  }

  const double ds = (m_data.s.back() - m_data.s.front()) /
                    static_cast<double>(m_data.s.size() - 1);
  m_data.kappa_s = smooth_kappa_closed_loop(m_data.kappa, ds);
  m_length = ds * static_cast<double>(m_data.s.size());
  m_index = std::make_unique<Index>(m_data);
  return true;
}

int TrackHandler::nearest_index(double x, double y) const {
  if (!m_index) {
    return -1;
  }
  std::size_t ret_index = 0;
  double out_dist_sqr = 0.0;
  const double query_pt[2] = {x, y};
  nanoflann::KNNResultSet<double> result_set(1);
  result_set.init(&ret_index, &out_dist_sqr);
  // Use exact nearest-neighbour search for reliable wall orientation.
  m_index->tree.findNeighbors(result_set, query_pt,
                              nanoflann::SearchParameters());
  return static_cast<int>(ret_index);
}

Frenet TrackHandler::to_frenet(double x, double y, double yaw) const {
  if (!m_index) {
    return {};
  }
  const int idx = nearest_index(x, y);
  if (idx < 0 || idx >= static_cast<int>(m_data.x.size())) {
    return {};
  }
  const double dx = x - m_data.x[idx];
  const double dy = y - m_data.y[idx];
  const double n = dx * m_data.n_x[idx] + dy * m_data.n_y[idx];
  double alpha = yaw - m_data.track_yaw[idx];
  while (alpha > kPi) {
    alpha -= 2 * kPi;
  }
  while (alpha < -kPi) {
    alpha += 2 * kPi;
  }
  return {m_data.s[idx], n, alpha};
}

double TrackHandler::sample_at_s(const std::vector<double> &values,
                                 double s) const {
  const std::size_t n = values.size();
  if (n == 0) {
    return 0.0;
  }
  if (n == 1) {
    return values[0];
  }

  const double s0 = m_data.s.front();
  const double ds = (m_data.s.back() - s0) / static_cast<double>(n - 1);
  const double loop_length = ds * static_cast<double>(n);
  double wrapped = std::fmod(s - s0, loop_length);
  if (wrapped < 0.0) {
    wrapped += loop_length;
  }
  const double target = s0 + wrapped;

  // First station above target (what upper_bound returns): start where the
  // uniform spacing puts it and walk the few stations the rounding can be off.
  const double guess = wrapped / ds + 1.0;
  std::size_t hi =
      guess < static_cast<double>(n) ? static_cast<std::size_t>(guess) : n;
  while (hi > 0 && m_data.s[hi - 1] > target) {
    --hi;
  }
  while (hi < n && m_data.s[hi] <= target) {
    ++hi;
  }
  if (hi == 0) {
    return values[0];
  }
  if (hi >= n) {
    const double t = (target - m_data.s.back()) / ds;
    return values.back() + t * (values[0] - values.back());
  }
  const double seg = m_data.s[hi] - m_data.s[hi - 1];
  const double t = (target - m_data.s[hi - 1]) / seg;
  return values[hi - 1] + t * (values[hi] - values[hi - 1]);
}

double TrackHandler::kappa_s(double s) const {
  return sample_at_s(m_data.kappa_s, s);
}

double TrackHandler::width_left(double s) const {
  return sample_at_s(m_data.w_l, s);
}

double TrackHandler::width_right(double s) const {
  return sample_at_s(m_data.w_r, s);
}

} // namespace sf
