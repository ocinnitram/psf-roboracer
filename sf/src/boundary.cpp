// Nearest-wall lookup for the LiDAR wall mask. A nanoflann KD-tree over the
// mapped boundary points; the detector asks distance(x, y) per hit.

#include "boundary.h"

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "nanoflann.hpp"

namespace sf {

namespace {

struct PointCloudAdaptor {
    const std::vector<Point2D>& points;

    explicit PointCloudAdaptor(const std::vector<Point2D>& pts) : points(pts) {}

    std::size_t kdtree_get_point_count() const { return points.size(); }

    double kdtree_get_pt(const std::size_t idx, const std::size_t dim) const {
        if (dim == 0) {
            return points[idx].x;
        }
        return points[idx].y;
    }

    // nanoflann adaptor API requires this member template; returning false
    // asks nanoflann to compute the bounding box itself.
    template <class Bbox>
    bool kdtree_get_bbox(Bbox&) const {
        return false;
    }
};

// nanoflann's index type over PointCloudAdaptor.
using WallKdTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloudAdaptor>,
    PointCloudAdaptor,
    2>;

} // namespace

struct BoundaryIndex::Impl {
    std::vector<Point2D> points;
    std::unique_ptr<PointCloudAdaptor> adaptor;
    std::unique_ptr<WallKdTree> tree;
};

BoundaryIndex::BoundaryIndex(std::vector<Point2D> wall_points)
    : m_impl(std::make_unique<Impl>()) {
    m_impl->points = std::move(wall_points);
    if (m_impl->points.empty()) {
        return;
    }
    m_impl->adaptor = std::make_unique<PointCloudAdaptor>(m_impl->points);
    m_impl->tree = std::make_unique<WallKdTree>(2, *m_impl->adaptor);
    m_impl->tree->buildIndex();
}

BoundaryIndex::BoundaryIndex(BoundaryIndex&&) noexcept = default;
BoundaryIndex& BoundaryIndex::operator=(BoundaryIndex&&) noexcept = default;
BoundaryIndex::~BoundaryIndex() = default;

double BoundaryIndex::distance(double x, double y) const {
    if (!m_impl || !m_impl->tree) {
        return std::numeric_limits<double>::infinity();
    }
    const double query[2] = {x, y};
    std::size_t nearest_idx = 0;
    double dist_sq = 0.0;
    nanoflann::KNNResultSet<double> result(1);
    result.init(&nearest_idx, &dist_sq);
    m_impl->tree->findNeighbors(result, query, nanoflann::SearchParameters());
    return std::sqrt(dist_sq);
}

} // namespace sf
