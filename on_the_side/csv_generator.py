import pandas as pd

# Fixed parameters
base_resource = 10
scale_factor = 1

# Variable parameters
density_factors = [0.5, 1.0, 1.5, 2.0]
num_users_list = range(20, 160, 20)  # Users from 20 to 100 with a step of 20
resource_sharing_factors = {'Low': 0.2, 'Medium': 0.5, 'High': 0.9}
max_users = 100

# Function to calculate scaled resources
def calc_scaled_resource(base_resource, density_factor, num_users, max_users, resource_sharing_factor, scale_factor):
    # Calculate user density impact
    density_impact = (num_users / max_users) ** density_factor
    sharing_factor = 1 - (resource_sharing_factor * (1 - density_impact))
    scaled_resource = base_resource * (pow(num_users, scale_factor) * sharing_factor)
    max_linear_res = num_users * base_resource
    percentage_increase = (scaled_resource - base_resource) / base_resource * 100
    return scaled_resource, percentage_increase, max_linear_res

# Generate the data
data = []
for density_factor in density_factors:
    for num_users in num_users_list:
        for sharing_label, resource_sharing_factor in resource_sharing_factors.items():
            scaled_resource, percentage_increase, max_res = calc_scaled_resource(
                base_resource, density_factor, num_users, max_users, resource_sharing_factor, scale_factor
            )
            data.append({
                "Density Factor": density_factor,
                "Number of Users": num_users,
                "Resource Sharing Level": sharing_label,
                "Scaled Resource": scaled_resource,
                "Percentage Increase": percentage_increase,
                "Max Resources": max_res
            })

# Create a DataFrame and save it as CSV
df = pd.DataFrame(data)
df.to_csv('data.csv', index=False)