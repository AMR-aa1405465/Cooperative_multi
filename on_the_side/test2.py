# this will actually plot the effect of different dfensity impact alone on the scaled resources for 
# fixed resources ans scale factor 

import numpy as np
import matplotlib.pyplot as plt


def plot_density_factor_effect(base_resource, density_factor,num_users,max_users,resource_sharing_factor,scale_factor,):
    # Calculate user density impact
    density_impact = (num_users / max_users) ** density_factor
    sharing_factor = 1 - (resource_sharing_factor * (1 - density_impact))
    scaled_resource = base_resource * (
            pow(num_users, scale_factor) * sharing_factor
        )
    # before_mod = base_resource * (pow(num_users, scale_factor))
    percentage_increase = (scaled_resource - base_resource) / base_resource
    # percentage_increase = (scaled_resource - before_mod) / before_mod
    return scaled_resource, percentage_increase



#base_resource, density_factor,num_users,max_users,resource_sharing_factor,scale_factor
# Setup
users = 20 #np.arange(1, 101,10)
max_users = 100
base_resource = 10
resource_sharing_factor = 0.2
scale_factor = 1.0
density_factors = np.linspace(1,5,5)

# Plot 1: Density Factor Effect
plt.figure(figsize=(15, 5))
# plt.subplot(1, 2, 1)
x = []
y = []

for nn_users in range(10,150,20):
    x.clear()
    y.clear()
    for density_factor in density_factors:
        # density_impact = (users/100) ** df
        scaled_resource,percentage_increase = plot_density_factor_effect(
            base_resource,
              density_factor,
              nn_users,
              max_users,
              resource_sharing_factor,
              scale_factor
        )
        x.append(density_factor)
        y.append(percentage_increase)
        print(f'Density Factor = {density_factor}, Percentage Increase = {percentage_increase}')
    plt.plot(x, y, label=f'Ro Cap.={round(nn_users/max_users*100)}%',linestyle='dashed' if nn_users > 100 else 'solid')
    print("************ ********** *********")

plt.title('Effect of Density Factor with resource_sharing_factor='+str(resource_sharing_factor))
plt.xlabel('Density Factor')
plt.ylabel('Resource Scaling % Increase')
plt.legend(loc='upper left', ncol=3)
plt.grid(True)
plt.savefig('density_factor_effect_'+str(resource_sharing_factor)+'.png')
plt.show()


