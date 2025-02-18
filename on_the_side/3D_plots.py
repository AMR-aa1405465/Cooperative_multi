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

# Setup for 3D plots
users_range = np.linspace(10, 150, 30)
density_factors = np.linspace(1, 5, 30)
X, Y = np.meshgrid(density_factors, users_range)

# Create a grid of subplots
fig = plt.figure(figsize=(20, 15))
resource_sharing_factors = np.linspace(0.2, 0.9, 8)  # 8 values from 0.2 to 0.9

for idx, resource_sharing_factor in enumerate(resource_sharing_factors, 1):
    Z = np.zeros_like(X)
    
    # Calculate Z values for this resource sharing factor
    for i, nn_users in enumerate(users_range):
        for j, density_factor in enumerate(density_factors):
            scaled_resource, _ = plot_density_factor_effect(
                base_resource,
                density_factor,
                nn_users,
                max_users,
                resource_sharing_factor,
                scale_factor
            )
            Z[i, j] = scaled_resource
    
    # Create subplot
    ax = fig.add_subplot(2, 4, idx, projection='3d')
    
    # Plot the surface
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    
    # Customize the plot
    ax.set_xlabel('Density Factor')
    ax.set_ylabel('Number of Users')
    ax.set_zlabel('Scaled Resource')
    ax.set_title(f'RSF={resource_sharing_factor:.1f}')
    
    # Add a color bar
    fig.colorbar(surface, ax=ax, label='Scaled Resource')
    
    # Adjust the viewing angle
    ax.view_init(elev=30, azim=45)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.95)

# Save and show
plt.savefig('3d_resource_scaling_all_rsf.png', bbox_inches='tight', pad_inches=0.5)
plt.show()


