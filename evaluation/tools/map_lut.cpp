// Steering lookup table for the MAP controller (evaluation/spliner.py), from the identified model.
//
//   g++ -O2 -std=c++17 -I sf/include -I . -I sf/externals evaluation/tools/map_lut.cpp sf/src/sf_model.cpp \
//       -o /tmp/map_lut && /tmp/map_lut > evaluation/data/map_lut.csv
//
// ForzaETH's format (system_identification/steering_lookup/cfg/*_lookup_table.csv): first row 0 and the
// speeds, first column the steering commands, entries the steady-state lateral acceleration. The steering
// command goes through the identified transmission (gamma curve) to the axle angle, and the axle angle
// through the filter's steady-state cornering (Pacejka single track, sf_model.cpp) to a_y.
#include "sf_model.h"
#include "vehicle/params.h"

#include <cstdio>

int main() {
  const int n_v = 36, n_d = 60;
  std::printf("0");
  for (int j = 0; j < n_v; ++j) std::printf(",%.6f", 0.5 + 0.1 * j);  // 0.5 .. 4.0 m/s
  std::printf("\n");
  for (int i = 0; i < n_d; ++i) {
    const double cmd = sf::kSteerLockCmdRad * i / (n_d - 1);
    std::printf("%.6f", cmd);
    for (int j = 0; j < n_v; ++j) {
      const double axle = sf::axle_target_from_command(cmd, sf::kSteerDeltaThRad, sf::kSteerGamma);
      std::printf(",%.6f", sf::steady_state_cornering(axle, 0.5 + 0.1 * j).a_y);
    }
    std::printf("\n");
  }
}
