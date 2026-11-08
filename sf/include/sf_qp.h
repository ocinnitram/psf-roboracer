#pragma once

#include <vector>

// Minimum-norm point of a convex polygon given as half-planes a·y >= b in
// the plane. The projection of the origin onto a polygon is the origin, a
// point on one edge, or a vertex, so the exact solution is the closest
// feasible point among those finitely many candidates.

namespace sf {

struct HalfPlane {
  double a0;
  double a1;
  double b;
};

// Returns false when the polygon is empty.
bool min_norm_point(const std::vector<HalfPlane> &rows, double *y0, double *y1);

} // namespace sf
