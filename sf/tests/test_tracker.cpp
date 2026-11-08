// Test tracked box shape and position as visible faces change.

#include "obstacle_tracker.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char *what) {
  std::printf("  %s: %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) {
    ++g_failures;
  }
}

bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

// Distance between two yaws as rectangle orientations (modulo a quarter turn).
double quarter_turn_gap(double a, double b) {
  const double q = 0.5 * M_PI;
  const double d = std::fmod(std::abs(a - b), q);
  return std::min(d, q - d);
}

// Extent of a rectangle along the unit direction at `yaw`.
double extent_along(const sf::Rect &r, double yaw) {
  return r.length * std::abs(std::cos(yaw - r.yaw)) +
         r.width * std::abs(std::sin(yaw - r.yaw));
}

// Create a detection whose near face is at `face_x`.
sf::RawDetection detection(double face_x, double length, double width,
                           bool single_side) {
  sf::RawDetection d;
  d.rect.cx = face_x + 0.5 * length;
  d.rect.cy = 0.0;
  d.rect.yaw = 0.0;
  d.rect.length = length;
  d.rect.width = width;
  d.single_side_estimate = single_side;
  return d;
}

std::vector<sf::Rect> feed(sf::ObstacleTracker *tracker,
                           const sf::RawDetection &det, const sf::Pose2D &pose,
                           int frames) {
  std::vector<sf::Rect> out;
  for (int k = 0; k < frames; ++k) {
    sf::RawDetections raw;
    raw.pose = pose;
    raw.valid = true;
    raw.items.push_back(det);
    out = tracker->update(raw);
  }
  return out;
}

} // namespace

int main() {
  const double kFaceX =
      4.875; // near face of a 0.35 (along) x 0.25 (across) box centred at 5.05
  const double kBias = sf::TrackedObstacle::kBiasAlongM;
  sf::ObstacleTracker tracker;

  // 1. A rear view reports a square while preserving the measured face.
  const sf::Pose2D behind{0.0, 0.0, 0.0};
  std::vector<sf::Rect> out =
      feed(&tracker, detection(kFaceX, 0.25, 0.25, true), behind, 4);
  check(out.size() == 1, "one face visible: one box reported");
  const sf::Rect &sq = out[0];
  check(near(sq.length, 0.25, 1e-9) && near(sq.width, 0.25, 1e-9),
        "one face visible: the box is the detector's square, no prior");
  check(near(sq.cx - 0.5 * sq.length, kFaceX + kBias, 1e-6),
        "one face visible: the measured face stays where the detector put it "
        "(plus the bias)");

  // 2. Alongside, both faces visible: the L-fit measures 0.35 along and
  //    0.25 across, and the box takes both.
  const sf::Pose2D beside{5.05, -3.0, 0.0};
  out = feed(&tracker, detection(kFaceX, 0.35, 0.25, false), beside, 4);
  check(out.size() == 1, "both faces visible: one box reported");
  check(near(out[0].length, 0.35, 1e-3) && near(out[0].width, 0.25, 1e-3),
        "both faces visible: both extents measured");

  // 3. Back to one face: the measured depth is kept and the centre sits
  //    half of it behind the face; the box is no longer a square.
  out = feed(&tracker, detection(kFaceX, 0.25, 0.25, true), behind, 8);
  const sf::Rect &kept = out[0];
  check(near(kept.length, 0.35, 1e-3) && near(kept.width, 0.25, 1e-3),
        "one face again: the measured depth is kept");
  check(near(kept.cx - 0.5 * kept.length, kFaceX + kBias, 0.02),
        "one face again: the face stays put with the centre half the depth "
        "behind it");

  // 4. A box turned 20 deg off the track, whose short side (0.25) runs along
  //    the fitted yaw. The track tangent (0 deg) is only the tracker's prior:
  //    the reported yaw follows the fit, and alternating two-face fits with
  //    one-face squares (the fit's length axis switching between the short
  //    and a guessed side) never turns the yaw by a quarter.
  const double kTurn = 20.0 * M_PI / 180.0;
  sf::ObstacleTracker turned;
  sf::RawDetection two_face;
  two_face.rect.cx = 5.05;
  two_face.rect.cy = 0.0;
  two_face.rect.yaw = kTurn;
  two_face.rect.length = 0.25;
  two_face.rect.width = 0.35;
  two_face.yaw_prior = 0.0;
  sf::RawDetection one_face = two_face;
  one_face.rect.length = 0.35;
  one_face.single_side_estimate = true;
  for (int k = 0; k < 10; ++k) {
    feed(&turned, two_face, beside, 1);
    out = feed(&turned, one_face, behind, 1);
  }
  check(out.size() == 1, "turned box: one box reported");
  check(quarter_turn_gap(out[0].yaw, kTurn) < 1.0 * M_PI / 180.0,
        "turned box: yaw follows the fit, not the track tangent prior");
  check(near(extent_along(out[0], kTurn), 0.25, 1e-2) &&
            near(extent_along(out[0], kTurn + 0.5 * M_PI), 0.35, 1e-2),
        "turned box: each extent stays on its own side of the box");

  std::printf("%s (%d failing checks)\n", g_failures ? "FAILED" : "OK",
              g_failures);
  return g_failures ? 1 : 0;
}
