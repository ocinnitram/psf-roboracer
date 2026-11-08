// Solve a two-variable closest-feasible-correction problem.

#include "sf_qp.h"

#include <cmath>
#include <vector>

namespace sf {

namespace {

// Check whether a correction satisfies every linear constraint.
bool satisfies_constraints(const std::vector<HalfPlane> &constraints,
                           double steering_correction,
                           double speed_correction) {
  for (const HalfPlane &constraint : constraints) {
    const double value =
        constraint.a0 * steering_correction +
        constraint.a1 * speed_correction;
    if (value < constraint.b -
                    1e-7 * (1.0 + std::abs(constraint.b))) {
      return false;
    }
  }
  return true;
}

} // namespace

// Check zero, each boundary projection, and each boundary intersection.
bool min_norm_point(
    const std::vector<HalfPlane> &constraints,
    double *steering_correction, double *speed_correction) {
  bool found = false;
  double best_squared_norm = 0.0;
  auto consider = [&](double steering, double speed) {
    const double squared_norm = steering * steering + speed * speed;
    if ((!found || squared_norm < best_squared_norm) &&
        satisfies_constraints(constraints, steering, speed)) {
      found = true;
      best_squared_norm = squared_norm;
      *steering_correction = steering;
      *speed_correction = speed;
    }
  };
  consider(0.0, 0.0);
  if (found) {
    return true;
  }
  for (std::size_t i = 0; i < constraints.size(); ++i) {
    const HalfPlane &first = constraints[i];
    const double coefficient_norm =
        first.a0 * first.a0 +
        first.a1 * first.a1;
    if (coefficient_norm > 0.0) {
      consider(first.b * first.a0 /
                   coefficient_norm,
               first.b * first.a1 /
                   coefficient_norm);
    }
    for (std::size_t j = i + 1; j < constraints.size(); ++j) {
      const HalfPlane &second = constraints[j];
      const double determinant =
          first.a0 * second.a1 -
          first.a1 * second.a0;
      const double determinant_scale =
          std::abs(first.a0 * second.a1) +
          std::abs(first.a1 * second.a0);
      if (std::abs(determinant) >
          1e-12 * (1.0 + determinant_scale)) {
        consider(
            (first.b * second.a1 -
             first.a1 * second.b) /
                determinant,
            (first.a0 * second.b -
             first.b * second.a0) /
                determinant);
      }
    }
  }
  return found;
}

} // namespace sf
