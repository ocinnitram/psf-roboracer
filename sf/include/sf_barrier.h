#pragma once

#include "sf_model.h"

// Collision-cone barrier and the terminal-set conditions. Pure functions of
// geometry.

namespace sf {

// Velocity heading of the cover disk at body offset l_i.
double disk_velocity_heading(const State &z, const SteadyStateCornering &c,
                             double l_i_m);

// Cone barrier h = |chi| - asin(r_keep / d) for a disk at (qx, qy) moving
// along heading phi, against a keep-out disk of radius r_keep centred at
// (cx, cy); chi is the bearing of the obstacle off the velocity. Negative
// once the disks overlap (asin saturates at pi/2).
double cone_h(double qx, double qy, double phi, double cx, double cy,
              double r_keep_m);

// Escape distance: the distance at which a lateral displacement `lateral_m`
// remains achievable at speed v with lateral budget a_lat, including the
// steering dead time.
double escape_distance(double v_mps, double lateral_m, double a_lat_mps2);

// Limit the heading so lateral motion can stop within the available room.
double lateral_stop_heading_limit(double a_perp_mps2, double room_m,
                                  double w_mps);

// Speed at which |kappa| v^2 spends the lateral budget, capped at the top
// speed the filter is configured for.
double curvature_speed_limit(double kappa, double a_lat_mps2,
                             double v_max_mps);

// Braking left on the friction circle after a lateral demand: lateral has
// priority, so at the budget no braking is allowed at all.
double brake_room(double a_y_mps2, double a_lat_mps2);

} // namespace sf
