#pragma once

#include "track.h"

#include <memory>
#include <vector>

namespace sf {

// Nearest-wall query over a cloud of mapped boundary points. Owns the points
// and the nanoflann index.
class BoundaryIndex {
public:
  // Takes ownership of wall_points. An empty cloud is valid: distance()
  // then returns +infinity (no wall nearby).
  explicit BoundaryIndex(std::vector<Point2D> wall_points);
  BoundaryIndex(const BoundaryIndex &) = delete;
  BoundaryIndex &operator=(const BoundaryIndex &) = delete;
  BoundaryIndex(BoundaryIndex &&) noexcept;
  BoundaryIndex &operator=(BoundaryIndex &&) noexcept;
  ~BoundaryIndex();

  // Euclidean distance in metres to the nearest wall point.
  double distance(double x, double y) const;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

} // namespace sf
