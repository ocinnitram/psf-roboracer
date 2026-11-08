"""Sensor noise the plant applies under sim2real; the detector is widened against it.

Two consumers, two unit systems:
- The policy observation (obs[0..3]) is noised in NORMALISED units: OBS_NOISE_S_NORM
  on s_norm, OBS_NOISE_YAW_RATE / OBS_NOISE_VEL / OBS_NOISE_YAW on the normalised
  channels. Training (ppo/rollout.py) reads these.
- The deploy pose the safety filter and detector localise with (rr_measured_state)
  is noised in SI: ODOM_NOISE_POS_M per axis and ODOM_NOISE_VEL_MPS. These are sensor
  specs in SI units, independent of the track. The detector's wall margin
  (stack.detector_config) is 2x the pose sigma.
LiDAR terms are metres.
"""

OBS_NOISE_S_NORM = 0.0006
OBS_NOISE_YAW_RATE = 0.015
OBS_NOISE_VEL = 0.0052
OBS_NOISE_YAW = 0.0042
ODOM_NOISE_POS_M = 0.014662390574812889
ODOM_NOISE_VEL_MPS = 0.0104
LIDAR_BIAS = 0.06
LIDAR_STOCHASTIC = 0.02
