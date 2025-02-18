from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import json
class RoomType(Enum):
    LIBRARY = "library"
    GALLERY = "gallery"
    GAME_ARENA = "game_arena"
    CLASSROOM = "classroom"
    CONFERENCE = "conference"

@dataclass
class ComputeRequirements:
    # Environment complexity
    polygon_count: int
    physics_objects: int
    interaction_points: int
    
    # Digital Twin requirements
    num_sensors: int
    state_variables: int
    update_frequency: float
    
    # User scaling parameters
    max_users: int
    user_scale_compute: float
    user_scale_bandwidth: float
    user_density_factor: float
    resource_sharing_factor: float
    
    # Quality parameters
    min_bitrate: float
    max_bitrate: float
    min_frame_rate: float
    max_frame_rate: float
    min_accuracy: float
    base_rotation_speed: float

@dataclass
class ResourceCosts:
    compute_cost_per_unit: float = 0.1    # $ per compute unit per hour
    bandwidth_cost_per_mbps: float = 0.05  # $ per Mbps per hour
    storage_cost_per_gb: float = 0.02      # $ per GB per hour

class VirtualRoomResources:
    def __init__(self, room_type: RoomType):
        self.room_type = room_type
        self.costs = ResourceCosts()
        
        # SSIM and VMAF coefficients
        self.ssim_coeff = {
            'b0': 0.65, 'b1': 0.368, 'b2': 1.23e-3,
            'b3': 0.850, 'b4': 1.23e-3
        }
        self.vmaf_coeff = {
            'b1': 36.13, 'b2': -1.66e-2,
            'b3': 11.62, 'b4': -6.07e-3
        }
        
        # Initialize room-specific requirements
        self.requirements = {
            RoomType.LIBRARY: ComputeRequirements(
                # Environment complexity
                polygon_count=200_000,
                physics_objects=50,
                interaction_points=5,
                
                # Digital Twin requirements
                num_sensors=50,
                state_variables=30,
                update_frequency=2,
                
                # User scaling
                max_users=50,
                user_scale_compute=0.7,
                user_scale_bandwidth=0.9,
                user_density_factor=0.8,
                resource_sharing_factor=0.6,
                
                # Quality parameters
                min_bitrate=20,
                max_bitrate=25,
                min_frame_rate=30,
                max_frame_rate=60,
                min_accuracy=0.8,
                base_rotation_speed=400
            ),
            
            RoomType.GAME_ARENA: ComputeRequirements(
                # Environment complexity
                polygon_count=500_000,
                physics_objects=200,
                interaction_points=30,
                
                # Digital Twin requirements
                num_sensors=200,
                state_variables=100,
                update_frequency=10,
                
                # User scaling
                max_users=100,
                user_scale_compute=0.9,
                user_scale_bandwidth=0.95,
                user_density_factor=0.9,
                resource_sharing_factor=0.3,
                
                # Quality parameters
                min_bitrate=35,
                max_bitrate=38,
                min_frame_rate=90,
                max_frame_rate=120,
                min_accuracy=0.65,
                base_rotation_speed=720
            )
        }
        
        self.req = self.requirements[room_type]

    def calculate_base_compute(self) -> Dict[str, float]:
        """Calculate base compute units needed for VE and DT"""
        ve_compute = (
            self.req.polygon_count/100_000 +
            self.req.physics_objects/100 +
            self.req.interaction_points/10
        )
        
        dt_compute = (
            self.req.num_sensors/100 +
            self.req.state_variables/100 +
            self.req.update_frequency/10
        )
        print(f've_compute: {ve_compute}, dt_compute: {dt_compute}')
        
        return {
            've_compute': ve_compute,
            'dt_compute': dt_compute
        }

    def calculate_ssim(self, bitrate: float, rotation_speed: float) -> float:
        """Calculate SSIM score"""
        b = self.ssim_coeff
        power = -(b['b3'] + b['b4'] * rotation_speed)
        ssim = 1 - (b['b1'] + b['b2'] * rotation_speed) * (bitrate ** power)
        return max(b['b0'], min(1.0, ssim))

    def calculate_vmaf(self, bitrate: float, rotation_speed: float) -> float:
        """Calculate VMAF score"""
        b = self.vmaf_coeff
        vmaf = b['b1'] + b['b2'] * rotation_speed + b['b3'] * bitrate + b['b4'] * rotation_speed * bitrate
        return max(0, min(100, vmaf))

    def calculate_user_scaled_resources(self, 
                                     base_resource: float, 
                                     num_users: int, 
                                     resource_type: str) -> float:
        """Calculate resource needs with user scaling"""
        req = self.req
        
        # Get appropriate scaling factor
        if resource_type == 'compute':
            scale_factor = req.user_scale_compute
        else:  # bandwidth
            scale_factor = req.user_scale_bandwidth
        
        # Calculate user density impact
        density_impact = (num_users / req.max_users) ** req.user_density_factor
        
        # Calculate shared resource factor
        sharing_factor = 1 - (req.resource_sharing_factor * (1 - density_impact))
        
        # Scale resources with user count
        scaled_resource = base_resource * (
            pow(num_users, scale_factor) * sharing_factor
        )
        
        return scaled_resource

    def calculate_immersiveness_score(self,
                                    bitrate: float,
                                    frame_rate: float,
                                    dt_accuracy: float,
                                    rotation_speed: float) -> float:
        """Calculate overall immersiveness score"""
        # Calculate quality metrics
        ssim = self.calculate_ssim(bitrate, rotation_speed)
        vmaf = self.calculate_vmaf(bitrate, rotation_speed) / 100  # Normalize to 0-1
        
        # Calculate normalized parameters
        norm_frame_rate = (frame_rate - self.req.min_frame_rate) / (self.req.max_frame_rate - self.req.min_frame_rate)
        
        # Weights for different components
        w1, w2, w3 = 0.4, 0.3, 0.3
        
        # Calculate immersiveness
        immersiveness = (
            w1 * ((ssim + vmaf) / 2) +  # Visual quality
            w2 * norm_frame_rate +       # Smoothness
            w3 * dt_accuracy            # Digital twin accuracy
        )
        
        return immersiveness

    def calculate_resources_and_costs(self,
                                    num_users: int,
                                    bitrate: float,
                                    frame_rate: float,
                                    dt_accuracy: float,
                                    rotation_speed: float,
                                    duration_hours: float = 1.0) -> Dict:
        """Calculate all resources and costs with user scaling"""
         # # Create visualization
    # plt.figure(figsize=(15, 10))
    
    # # Plot cost per user
    # plt.subplot(2, 2, 1)
    # for room_type in rooms:
    #     costs = [r['total']['cost_per_user'] for r in results[room_type]]
    #     plt.plot(user_counts[:len(costs)], costs, label=room_type.value)
    # plt.title('Cost per User vs. Number of Users')
    # plt.xlabel('Number of Users')
    # plt.ylabel('Cost per User ($)')
    # plt.legend()
    
    # # Plot compute efficiency
    # plt.subplot(2, 2, 2)
    # for room_type in rooms:
    #     efficiency = [r['scaling_metrics']['compute_efficiency'] for r in results[room_type]]
    #     plt.plot(user_counts[:len(efficiency)], efficiency, label=room_type.value)
    # plt.title('Compute Efficiency vs. Number of Users')
    # plt.xlabel('Number of Users')
    # plt.ylabel('Compute Units per User')
    # plt.legend()
    
    # # Plot bandwidth efficiency
    # plt.subplot(2, 2, 3)
    # for room_type in rooms:
    #     efficiency = [r['scaling_metrics']['bandwidth_efficiency'] for r in results[room_type]]
    #     plt.plot(user_counts[:len(efficiency)], efficiency, label=room_type.value)
    # plt.title('Bandwidth Efficiency vs. Number of Users')
    # plt.xlabel('Number of Users') # # Create visualization
    # plt.figure(figsize=(15, 10))
    
    # # Plot cost per user
    # plt.subplot(2, 2, 1)
    # for room_type in rooms:
    #     costs = [r['total']['cost_per_user'] for r in results[room_type]]
    #     plt.plot(user_counts[:len(costs)], costs, label=room_type.value)
    # plt.title('Cost per User vs. Number of Users')
    # plt.xlabel('Number of Users')
    # plt.ylabel('Cost per User ($)')
    # plt.legend()
    
    # # Plot compute efficiency
    # plt.subplot(2, 2, 2)
    # for room_type in rooms:
    #     efficiency = [r['scaling_metrics']['compute_efficiency'] for r in results[room_type]]
    #     plt.plot(user_counts[:len(efficiency)], efficiency, label=room_type.value)
    # plt.title('Compute Efficiency vs. Number of Users')
    # plt.xlabel('Number of Users')
    # plt.ylabel('Compute Units per User')
    # plt.legend()
    
    # # Plot bandwidth efficiency
    # plt.subplot(2, 2, 3)
    # for room_type in rooms:
    #     efficiency = [r['scaling_metrics']['bandwidth_efficiency'] for r in results[room_type]]
    #     plt.plot(user_counts[:len(efficiency)], efficiency, label=room_type.value)
    # plt.title('Bandwidth Efficiency vs. Number of Users')
    # plt.xlabel('Number of Users')
    # plt.ylabel('Bandwidth per User (Mbps)')
    # plt.legend()
    
    # # Plot immersiveness
    # plt.subplot(2, 2, 4)
    # for room_type in rooms:
    #     immersiveness = [r['quality_metrics']['immersiveness'] for r in results[room_type]]
    #     plt.plot(user_counts[:len(immersiveness)], immersiveness, label=room_type.value)
    # plt.title('Immersiveness vs. Number of Users')
    # plt.xlabel('Number of Users')
    # plt.ylabel('Immersiveness Score')
    # plt.legend()
    
    # plt.tight_layout()
    # plt.show()
    # plt.ylabel('Bandwidth per User (Mbps)')
    # plt.legend()
    
    # # Plot immersiveness
    # plt.subplot(2, 2, 4)
    # for room_type in rooms:
    #     immersiveness = [r['quality_metrics']['immersiveness'] for r in results[room_type]]
    #     plt.plot(user_counts[:len(immersiveness)], immersiveness, label=room_type.value)
    # plt.title('Immersiveness vs. Number of Users')
    # plt.xlabel('Number of Users')
    # plt.ylabel('Immersiveness Score')
    # plt.legend()
    
    # plt.tight_layout()
    # plt.show()
        # Input validation
        if num_users > self.req.max_users:
            raise ValueError(f"Number of users ({num_users}) exceeds room capacity ({self.req.max_users})")
        
        # Get base compute requirements (how many units of compute are needed for VE and DT)
        base_compute = self.calculate_base_compute()
        print(f'base_compute: {base_compute}')
        # exit(0)
        
        # Frame rate scaling factors
        compute_scale = pow(frame_rate/self.req.min_frame_rate, 1.1)  # Slightly superlinear
        bandwidth_scale = pow(frame_rate/self.req.min_frame_rate, 0.5)  # Square root relationship
        
        # Calculate VE resources
        ve_compute = self.calculate_user_scaled_resources(
            base_compute['ve_compute'] * compute_scale,
            num_users,
            'compute'
        ) * duration_hours


        print("before scaling=",base_compute['ve_compute']," after scaling with frame rate and number of users =",ve_compute, 
              " percentage increase=", (ve_compute - base_compute['ve_compute'])/base_compute['ve_compute'], " with num_users=", num_users)
        # exit(0)
        
        
        # Calculate DT resources
        dt_compute = self.calculate_user_scaled_resources(
            base_compute['dt_compute'] * dt_accuracy * compute_scale,
            num_users,
            'compute'
        ) * duration_hours
        
        print("before scaling=",base_compute['dt_compute']," after scaling with frame rate =",frame_rate," and number of users =",
              dt_compute, " percentage increase=", (dt_compute - base_compute['dt_compute'])/base_compute['dt_compute'], " with num_users=", num_users)
        

        ve_bandwidth = self.calculate_user_scaled_resources(
            bitrate * bandwidth_scale,
            num_users,
            'bandwidth'
        ) * duration_hours
        
        dt_bandwidth = self.calculate_user_scaled_resources(
            0.1 * bitrate * dt_accuracy * bandwidth_scale,
            num_users,
            'bandwidth'
        ) * duration_hours
        print("ve_bandwidth=",ve_bandwidth," dt_bandwidth=",dt_bandwidth)
        # exit(0)
        
        # Calculate costs
        ve_compute_cost = ve_compute * self.costs.compute_cost_per_unit
        ve_bandwidth_cost = ve_bandwidth * self.costs.bandwidth_cost_per_mbps
        dt_compute_cost = dt_compute * self.costs.compute_cost_per_unit
        dt_bandwidth_cost = dt_bandwidth * self.costs.bandwidth_cost_per_mbps
        # here we ca

        # Calculate immersiveness
        immersiveness = self.calculate_immersiveness_score(
            bitrate, frame_rate, dt_accuracy, rotation_speed
        )
        
        total_cost = ve_compute_cost + ve_bandwidth_cost + dt_compute_cost + dt_bandwidth_cost
        
        a = {
            'quality_metrics': {
                'ssim': self.calculate_ssim(bitrate, rotation_speed),
                'vmaf': self.calculate_vmaf(bitrate, rotation_speed),
                'immersiveness': immersiveness
            },
            'virtual_environment': {
                'compute_units': ve_compute,
                'bandwidth_mbps': ve_bandwidth,
                'compute_cost': ve_compute_cost,
                'bandwidth_cost': ve_bandwidth_cost
            },
            'digital_twin': {
                'compute_units': dt_compute,
                'bandwidth_mbps': dt_bandwidth,
                'compute_cost': dt_compute_cost,
                'bandwidth_cost': dt_bandwidth_cost
            },
            'total': {
                'cost': total_cost,
                'cost_per_user': total_cost / num_users,
                'compute_units': ve_compute + dt_compute,
                'bandwidth_mbps': ve_bandwidth + dt_bandwidth
            },
            'scaling_metrics': {
                'user_density': num_users / self.req.max_users,
                'compute_efficiency': (ve_compute + dt_compute) / (num_users * duration_hours),
                'bandwidth_efficiency': (ve_bandwidth + dt_bandwidth) / (num_users * duration_hours)
            }
            
        }
        # print(json.dumps(a, indent=4))
        return a

def test_and_visualize():
    # Create test scenarios
    # rooms = [RoomType.LIBRARY, RoomType.GAME_ARENA]
    rooms = [RoomType.LIBRARY]
    user_counts = [5]#, 10, 20, 30, 40, 50]
    results = {}
    
    for room_type in rooms:
        room = VirtualRoomResources(room_type)
        room_results = []
        
        for num_users in user_counts:
            try:
                result = room.calculate_resources_and_costs(
                    num_users=num_users,
                    bitrate=room.req.min_bitrate,
                    frame_rate=room.req.min_frame_rate,
                    dt_accuracy=room.req.min_accuracy,
                    rotation_speed=room.req.base_rotation_speed,
                    duration_hours=1.0
                )
                import json
                print(json.dumps({"user_count": num_users, "result": result}, indent=4))
                room_results.append(result)
            except ValueError as e:
                print(f"Error for {room_type.value} with {num_users} users: {e}")
        
        results[room_type] = room_results
    
    # # Create visualization
    # plt.figure(figsize=(15, 10))
    
    # # Plot cost per user
    # plt.subplot(2, 2, 1)
    # for room_type in rooms:
    #     costs = [r['total']['cost_per_user'] for r in results[room_type]]
    #     plt.plot(user_counts[:len(costs)], costs, label=room_type.value)
    # plt.title('Cost per User vs. Number of Users')
    # plt.xlabel('Number of Users')
    # plt.ylabel('Cost per User ($)')
    # plt.legend()
    
    # # Plot compute efficiency
    # plt.subplot(2, 2, 2)
    # for room_type in rooms:
    #     efficiency = [r['scaling_metrics']['compute_efficiency'] for r in results[room_type]]
    #     plt.plot(user_counts[:len(efficiency)], efficiency, label=room_type.value)
    # plt.title('Compute Efficiency vs. Number of Users')
    # plt.xlabel('Number of Users')
    # plt.ylabel('Compute Units per User')
    # plt.legend()
    
    # # Plot bandwidth efficiency
    # plt.subplot(2, 2, 3)
    # for room_type in rooms:
    #     efficiency = [r['scaling_metrics']['bandwidth_efficiency'] for r in results[room_type]]
    #     plt.plot(user_counts[:len(efficiency)], efficiency, label=room_type.value)
    # plt.title('Bandwidth Efficiency vs. Number of Users')
    # plt.xlabel('Number of Users')
    # plt.ylabel('Bandwidth per User (Mbps)')
    # plt.legend()
    
    # # Plot immersiveness
    # plt.subplot(2, 2, 4)
    # for room_type in rooms:
    #     immersiveness = [r['quality_metrics']['immersiveness'] for r in results[room_type]]
    #     plt.plot(user_counts[:len(immersiveness)], immersiveness, label=room_type.value)
    # plt.title('Immersiveness vs. Number of Users')
    # plt.xlabel('Number of Users')
    # plt.ylabel('Immersiveness Score')
    # plt.legend()
    
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    test_and_visualize()
