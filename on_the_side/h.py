import matplotlib.pyplot as plt
import numpy as np


def calc_scaled_resource(base_resource, density_factor,num_users,max_users,resource_sharing_factor,scale_factor,):
    # Calculate user density impact
    density_impact = (num_users / max_users) ** density_factor
    sharing_factor = 1 - (resource_sharing_factor * (1 - density_impact))
    scaled_resource = base_resource * (
            pow(num_users, scale_factor) * sharing_factor
        )
    max_res=num_users*base_resource
    # before_mod = base_resource * (pow(num_users, scale_factor))
    percentage_increase = (scaled_resource - base_resource) / base_resource * 100
    # percentage_increase = (scaled_resource - before_mod) / before_mod
    return scaled_resource, percentage_increase,max_res


density_factor = 1
resource_sharing_factor = 0.2
scale_factor = 1
base_resource = 2
num_users = 5
max_users = 100
density_impact = (num_users / max_users) ** density_factor



scaled_resource, percentage_increase,max_res = calc_scaled_resource(base_resource, density_factor,num_users,max_users,resource_sharing_factor,scale_factor)
print("Started with ",base_resource,"", "--> (",scaled_resource, ",",max_res, ") with a ",percentage_increase,"% increase")
# print("Percentage increase is ",percentage_increase)
