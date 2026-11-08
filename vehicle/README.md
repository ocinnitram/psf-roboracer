# Vehicle parameters

`params.h` is the one definition of the car: mass, geometry, tyre, actuator lags,
limits, sensor mount. Each number is either measured on the car or derived from
measured values.

`gen_params.py` writes `params.yaml` from it at build time (`./rebuild`, or the
`vehicle_params` CMake target in `sf/`). The filter's model includes the header; the
sim's Python side and the vehicle's ROS 2 node (not part of this repository) read the
yaml.

`tests/test_params.py` checks that the generator is deterministic and spot-checks
values in the generated yaml.
