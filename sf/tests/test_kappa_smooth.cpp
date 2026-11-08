// Smoothed kappa_s(s) and width interpolators.

#include "vehicle/params.h"
#include "track.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace sf;

namespace {

constexpr const char *kTrackPath = "tracks/lab-20260906/raceline.csv";

constexpr double kRelTol = 0.10;
// Top speed of the lab vehicle.
constexpr double kVMaxMps = 4.0;

int g_failures = 0;

void check(bool ok, const std::string &what) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) {
    ++g_failures;
  }
}

void check_rel(double got, double want, double tol, const std::string &what) {
  const bool ok = std::abs(got - want) <= tol * std::abs(want);
  std::printf("  [%s] %s (got %.4f, want %.4f)\n", ok ? "PASS" : "FAIL",
              what.c_str(), got, want);
  if (!ok) {
    ++g_failures;
  }
}

double speed_limit(double kappa) {
  return std::sqrt(4.121 / std::max(std::abs(kappa), 1e-9));
}

struct AbsStats {
  double max;
  double median;
};

AbsStats abs_stats(const std::vector<double> &kappa, const char *label) {
  std::vector<double> abs_kappa;
  abs_kappa.reserve(kappa.size());
  for (double k : kappa) {
    abs_kappa.push_back(std::abs(k));
  }
  std::sort(abs_kappa.begin(), abs_kappa.end());
  const AbsStats stats{abs_kappa.back(), abs_kappa[abs_kappa.size() / 2]};
  std::printf("    %s: max|k| %.4f, median|k| %.4f, n %zu\n", label, stats.max,
              stats.median, abs_kappa.size());
  return stats;
}

void test_spec_stats_on_racing_line() {
  std::printf("curvature stats on tracks/lab-20260906/raceline.csv\n");
  TrackHandler track;
  check(track.load(kTrackPath), "repo racing-line CSV loads");
  const TrackData &data = track.data();
  check(data.kappa.size() == data.kappa_s.size(),
        "kappa_s allocated same length as kappa");

  const AbsStats raw = abs_stats(data.kappa, "raw kappa  ");
  const AbsStats smoothed = abs_stats(data.kappa_s, "smoothed   ");
  check_rel(smoothed.max, raw.max, kRelTol,
            "smoothed max|kappa_s| keeps the raw maximum (10%)");
  check_rel(smoothed.median, raw.median, kRelTol,
            "smoothed median|kappa_s| keeps the raw median (10%)");
  check(smoothed.max <= raw.max + 1e-12, "smoothing never raises the raw peak");

  int capped = 0;
  for (double k : data.kappa_s) {
    if (speed_limit(k) < kVMaxMps) {
      ++capped;
    }
  }
  check(capped > 0, "corridor speed caps some racing-line stations");
}

void test_interpolators_finite() {
  std::printf("interpolators finite across the loop\n");
  TrackHandler track;
  check(track.load(kTrackPath), "repo racing-line CSV loads");
  const TrackData &data = track.data();
  const double loop_length =
      data.s.back() +
      (data.s.back() - data.s.front()) / static_cast<double>(data.s.size() - 1);
  const double probes[3] = {0.0, loop_length / 2.0, loop_length};
  bool all_finite = true;
  for (double s : probes) {
    all_finite = all_finite && std::isfinite(track.kappa_s(s));
    all_finite = all_finite && std::isfinite(track.width_left(s));
    all_finite = all_finite && std::isfinite(track.width_right(s));
  }
  check(all_finite,
        "kappa_s/width_left/width_right finite at s=0, mid, length");

  check_rel(track.kappa_s(0.0), track.kappa_s(loop_length), 1e-9,
            "kappa_s is periodic with the loop length");
}

void test_synthetic_zero_kappa_stays_zero() {
  std::printf("synthetic straight track: kappa=0 stays 0\n");
  const std::string straight_path =
      (std::filesystem::temp_directory_path() / "kappa_smooth_straight.csv")
          .string();
  {
    std::ofstream file(straight_path);
    file << "x,y,s,s_norm,track_yaw_rad,kappa,w_l,w_r,n_x,n_y\n";
    constexpr int kPoints = 100;
    constexpr double kSpacingM = 0.05;
    for (int i = 0; i < kPoints; ++i) {
      const double s = i * kSpacingM;
      file << s << ",0," << s << ',' << s / ((kPoints - 1) * kSpacingM)
           << ",0,0,0.8,1.2,0,1\n";
    }
  }
  TrackHandler track;
  check(track.load(straight_path), "synthetic straight track loads");
  const double probes[3] = {0.0, 2.0, 4.9};
  bool zero = true;
  for (double s : probes) {
    zero = zero && track.kappa_s(s) == 0.0;
  }
  check(zero, "kappa_s is exactly 0 on a straight track");
  check_rel(track.width_left(0.0), 0.8, 1e-9, "width_left interpolates w_l");
  check_rel(track.width_right(4.9), 1.2, 1e-9, "width_right interpolates w_r");
  std::remove(straight_path.c_str());
}

} // namespace

int main() {
  test_spec_stats_on_racing_line();
  test_interpolators_finite();
  test_synthetic_zero_kappa_stays_zero();

  if (g_failures == 0) {
    std::printf("\nALL PASS\n");
    return 0;
  }
  std::printf("\n%d FAILURE(S)\n", g_failures);
  return 1;
}
