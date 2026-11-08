// Closest feasible correction under linear constraints.
#include "sf_qp.h"

#include <cmath>
#include <cstdio>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    failures += ok ? 0 : 1;
}
bool near(double a, double b) { return std::abs(a - b) < 1e-9; }
} // namespace

int main() {
    double y0 = 0.0;
    double y1 = 0.0;
    check(sf::min_norm_point(
              {{1.0, 0.0, -1.0}, {0.0, 1.0, -1.0}}, &y0, &y1) && near(y0, 0.0) &&
              near(y1, 0.0),
          "origin inside the polygon is returned");
    check(sf::min_norm_point(
              {{1.0, 0.0, 2.0}, {0.0, 1.0, -5.0}}, &y0, &y1) && near(y0, 2.0) &&
              near(y1, 0.0),
          "projection onto a single edge");
    check(sf::min_norm_point(
              {{1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}}, &y0, &y1) && near(y0, 1.0) &&
              near(y1, 1.0),
          "projection onto a vertex");
    check(sf::min_norm_point(
              {{1.0, 1.0, 2.0}}, &y0, &y1) && near(y0, 1.0) && near(y1, 1.0),
          "projection onto an oblique edge");
    check(sf::min_norm_point(
              {{1.0, 1.0, 2.0}, {-1.0, 0.0, -0.5}}, &y0, &y1) && near(y0, 0.5) &&
              near(y1, 1.5),
          "box constraint moves the solution to a vertex");
    check(!sf::min_norm_point(
              {{1.0, 0.0, 1.0}, {-1.0, 0.0, 0.0}}, &y0, &y1),
          "empty polygon is reported");
    std::printf(failures ? "FAILED (%d)\n" : "OK (0 failing checks)\n", failures);
    return failures ? 1 : 0;
}
