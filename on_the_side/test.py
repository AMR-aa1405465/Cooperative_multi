# import matplotlib.pyplot as plt
# import numpy as np

# def calculate_user_density_impact(num_users, max_users, user_density_factor):
#     return (num_users / max_users) ** user_density_factor

# # Constants
# base_resource = 100
# num_users = np.arange(1, 101)  # Varying number of users from 1 to 100
# max_users = 100
# user_density_factors = [0.5, 1.0, 1.5, 2.0]  # Different density factors to test

# # Plotting
# plt.figure(figsize=(10, 6))
# for udf in user_density_factors:
#     density_impact = calculate_user_density_impact(num_users, max_users, udf)
#     scaled_resources = base_resource * density_impact
#     plt.plot(num_users, scaled_resources, label=f'User Density Factor: {udf}')

# plt.title('Effect of User Density Factor on Resource Scaling')
# plt.xlabel('Number of Users')
# plt.ylabel('Scaled Resources')
# plt.legend()
# plt.grid(True)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_both_factors():
    # Setup
    users = np.arange(1, 101)
    base_resource = 100
    
    # Plot 1: Density Factor Effect
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    
    density_factors = [0.5, 1.0, 1.5, 2.0]
    for df in density_factors:
        density_impact = (users/100) ** df
        plt.plot(users, density_impact * base_resource, 
                label=f'Density Factor={df}')
    
    plt.title('Effect of Density Factor')
    plt.xlabel('Number of Users')
    plt.ylabel('Resource Needed')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Sharing Factor Effect
    plt.subplot(1, 2, 2)
    
    sharing_factors = [0.2, 0.4, 0.6, 0.8]
    density_impact = (users/100)  # fixed density factor of 1.0
    
    for sf in sharing_factors:
        sharing_impact = 1 - (sf * (1 - density_impact))
        plt.plot(users, sharing_impact * base_resource, 
                label=f'Sharing Factor={sf}')
    
    plt.title('Effect of Sharing Factor')
    plt.xlabel('Number of Users')
    plt.ylabel('Resource Needed')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_both_factors()