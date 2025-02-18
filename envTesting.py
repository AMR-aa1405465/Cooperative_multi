import sys
import os
import model.VirtualRoom as vr

# sys.path.append('/Users/mac/Documents/Cooperative_game/')
from helpers.Constants import BEHAVIORAL_ACCURACY_WEIGHT, RESPONSIVE_WEIGHT, VISUAL_QUALITY_WEIGHT

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from helpers.Constants import BEHAVIORAL_ACCURACY_WEIGHT, RESPONSIVE_WEIGHT, VISUAL_QUALITY_WEIGHT
# print("calculate metrics testing..")
library_room = vr.VirtualRoom(id=1,
                              min_bitrate=100,
                              max_bitrate=1000,
                              min_frame_rate=30,
                              max_frame_rate=120,
                              min_structural_accuracy=0.5,
                              max_structural_accuracy=1,
                              rotation_speed=360,
                              quality_weights={'ssim': 0.5, 'vmaf': 0.5},
                              base_compute_units=100,
                              user_scaling_factor=0.5,
                              unit_bandwidth_cost=0.01,
                              unit_compute_cost=0.01,
                              min_behavioral_accuracy=0.5,
                              max_behavioral_accuracy=1,
                              max_users=100,
                              user_scale_compute=0.5,
                              user_scale_bandwidth=0.5,
                              user_density_factor=0.5,
                              resource_sharing_factor=0.5,
                              polygon_count=200000,
                              physics_objects=50,
                              interaction_points=5,
                              num_sensors=50,
                              state_variables=30,
                              update_frequency=2)

print(library_room.calculate_metrics(given_bitrate=100,
                                     num_heads_clients=10,
                                     behavioral_accuracy=0.9,
                                     given_frame_rate=80))
