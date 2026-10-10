# Runtime config accepted by the Explore skill.
#
# Package-level `config:` fields choose how the skill explores. Per-request
# limits do not belong here; they are fields of each
# robonix/skill/explore/explore request:
#
# - area_hint: string describing the requested exploration area.
# - timeout_s: unsigned integer execution deadline in seconds.
# - max_speed_m_s: floating-point linear speed limit in metres per second.
#
# This file is documentation only and is not loaded by the provider.

config:
  # Which frontier to visit next (explore_skill/strategies.py):
  #   nearest          explore_lite's cost (3 x path length - frontier
  #                    length), with path length searched on the map.
  #   mrtsp            the MRTSP start cost from frontier_exploration_ros2.
  #   frontier_greedy  the original scorer: size over straight-line
  #                    distance, goal on the straight line to the robot.
  strategy: nearest
  # Radius of the circle the robot's centre must keep clear of mapped
  # obstacles, in metres. Lite3: 0.235 (half the 0.47 m padded width).
  # Default: 0.22.
  robot_radius_m: 0.235
