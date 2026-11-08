// Test the SafetyFilter facade: design-speed validation and the policy speed cap.

#include "sf.h"

#include <cstdio>
#include <limits>
#include <stdexcept>

using namespace sf;

namespace {

constexpr const char* kTrackPath = "tracks/lab-20260906/raceline.csv";
// Top speed of the lab vehicle.
constexpr double kVMaxMps = 4.0;
// Speed of the test pose, and the request step either side of v_max_mps.
constexpr double kPoseSpeedMps = 3.0;
constexpr double kRequestStepMps = 1.0;
int failures = 0;

void check(bool condition, const char* message) {
    std::printf("  [%s] %s\n", condition ? "PASS" : "FAIL", message);
    if (!condition) {
        ++failures;
    }
}

// True when construction throws std::invalid_argument; any other exception escapes.
bool rejects_v_max(const TrackHandler& track, double v_max_mps) {
    try {
        const SafetyFilter filter(track, v_max_mps);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

// Certify a straight-ahead request at `v_mps` from the first station, fresh filter.
SafetyFilterOutput certify_fresh(const TrackHandler& track, double v_mps) {
    const TrackData& d = track.data();
    VehicleState state;
    state.x = d.x[0];
    state.y = d.y[0];
    state.yaw = d.track_yaw[0];
    state.v = kPoseSpeedMps;
    Action request;
    request.v = v_mps;
    SafetyFilter filter(track, kVMaxMps);
    return filter.certify(state, {}, request);
}

bool same_command(const SafetyFilterOutput& a, const SafetyFilterOutput& b) {
    return a.outcome == b.outcome && a.command.delta == b.command.delta &&
           a.command.v == b.command.v;
}

} // namespace

int main() {
    TrackHandler track;
    if (!track.load(kTrackPath)) {
        std::printf("cannot load %s (run with working directory sf/)\n", kTrackPath);
        return 1;
    }
    std::printf("== safety filter facade ==\n");

    check(rejects_v_max(track, std::numeric_limits<double>::quiet_NaN()),
          "NaN v_max_mps is rejected");
    check(rejects_v_max(track, std::numeric_limits<double>::infinity()),
          "infinite v_max_mps is rejected");
    check(rejects_v_max(track, 0.0), "zero v_max_mps is rejected");
    check(rejects_v_max(track, -kVMaxMps), "negative v_max_mps is rejected");
    check(!rejects_v_max(track, kVMaxMps), "the lab's v_max_mps is accepted");

    const SafetyFilterOutput above = certify_fresh(track, kVMaxMps + kRequestStepMps);
    const SafetyFilterOutput at = certify_fresh(track, kVMaxMps);
    const SafetyFilterOutput below = certify_fresh(track, kVMaxMps - kRequestStepMps);
    check(above.rl_speed_capped, "a request above v_max_mps is flagged");
    check(same_command(above, at), "a request above v_max_mps certifies as one at v_max_mps");
    check(!at.rl_speed_capped, "a request at v_max_mps is not flagged");
    check(!below.rl_speed_capped, "a request below v_max_mps is not flagged");

    std::printf("%s (%d failing checks)\n", failures ? "FAILURES" : "OK", failures);
    return failures ? 1 : 0;
}
