"""
This file contains the code for the VirtualRoom class.
The VirtualRoom class represent a virtual room in the metaverse.
Each virtual room accomdates multiple clients. 
Each virtual room has minimum and maximum bitrate required.
Each virtual room has minimum and maximum frame rate required.
Each virtual room has minimum and maximum beahvoirual accuracy required for its digital twin. 
Each room has a user scale factor. ( a value between 0 and 1 )
"""
import math
from typing import Dict, Tuple

from helpers.Constants import BEHAVIORAL_ACCURACY_WEIGHT, RESPONSIVE_WEIGHT, VISUAL_QUALITY_WEIGHT
import numpy as np
import sys
import os


class VirtualRoom:
    _virtual_room_id_counter = 0
    def __init__(self,
                 min_bitrate: float,
                 max_bitrate: float,
                 min_frame_rate: float,
                 max_frame_rate: float,
                 min_structural_accuracy: float,
                 max_structural_accuracy: float,
                 rotation_speed: float,  # fixed for now
                 quality_weights: Dict[str, float],  # for ssim and vmaf (different for each room type)
                #  base_compute_units: float,  # base compute resources needed
                 user_scaling_factor: float,  # how resources scale with users
                 unit_bandwidth_cost: float,  # cost per unit of bandwidth
                 unit_compute_cost: float,  # cost per unit of compute
                 min_behavioral_accuracy: float,  # for dt only
                 max_behavioral_accuracy: float,  # 1 for dt only
                 max_users: int,
                 user_scale_compute: float,
                 user_scale_bandwidth: float,
                 user_density_factor: float,  # still not decided about this
                 resource_sharing_factor: float,  # still not decided about this
                 polygon_count: int = 200_000,
                 physics_objects: int = 50,
                 interaction_points: int = 5,  # what happens to the dt & when it should be updated.
                 num_sensors: int = 50,  # number of sensors the dt needs to track
                 state_variables: int = 30,  # number of variables the dt needs to track
                 update_frequency: int = 2,  # for the dt
                 ):
        """
        The density factor describes how the num_users affects the scaled resources given the maximum capacity
        The sharing factor describes how well the resource can be shared (Low=0.2,Med=0.5,high=0.9)
        The scale factor, provides a general scaling independent of the maximum capacity
        """
        self.id: int = VirtualRoom._virtual_room_id_counter
        VirtualRoom._virtual_room_id_counter += 1
        self.max_users: int = max_users
        self.user_scale_compute: float = user_scale_compute  # general scaling factor for compute
        self.user_scale_bandwidth: float = user_scale_bandwidth  # general scaling factor for bandwidth
        self.user_density_factor: float = user_density_factor  # how number of users affects the scaled resources given the maximum capacity
        self.resource_sharing_factor: float = resource_sharing_factor  # how well the resource can be shared (Low=0.2,Med=0.5,high=0.9)

        # Environment complexity
        self.polygon_count = polygon_count
        self.physics_objects = physics_objects
        self.interaction_points = interaction_points

        # Digital Twin requirements
        self.num_sensors = num_sensors
        self.state_variables = state_variables
        self.update_frequency = update_frequency
        
        # Quality parameters
        self.min_bitrate: float = min_bitrate
        self.max_bitrate: float = max_bitrate
        self.min_frame_rate: float = min_frame_rate
        self.max_frame_rate: float = max_frame_rate
        self.min_structural_accuracy: float = min_structural_accuracy
        self.max_structural_accuracy: float = max_structural_accuracy
        self.rotation_speed = rotation_speed
        # Quality weights   
        self.quality_weights: Dict[str, float] = quality_weights  # for ssim and vmaf
        # Base compute and storage units
        self.base_compute_units: float = 0  # base compute resources needed
        # self.base_storage_units: float = base_storage_units  # base storage needed (GB)
        # User scaling factor
        self.user_scaling_factor: float = user_scaling_factor  # how resources scale with users
        # Behavioral accuracy weight
        self.BEH_ACC_WEIGHT: float = 0.2  # weight of the behavioral accuracy
        # Bandwidth and compute costs
        self.unit_bandwidth_cost: float = unit_bandwidth_cost  # cost per unit of bandwidth
        self.unit_compute_cost: float = unit_compute_cost  # cost per unit of compute
        self.min_behavioral_accuracy: float = min_behavioral_accuracy  # minimum behavioral accuracy required
        self.max_behavioral_accuracy: float = max_behavioral_accuracy  # maximum behavioral accuracy required
        self.ssim_coeff = {
            'b0': 0.65,
            'b1': 0.368,
            'b2': 1.23e-3,
            'b3': 0.850,
            'b4': 1.23e-3
        }
        # VMAF coefficients
        self.vmaf_coeff = {
            'b1': 36.13,
            'b2': -1.66e-2,
            'b3': 11.62,
            'b4': -6.07e-3
        }
        self.post_init_checker()

    def post_init_checker(self):
        # check if the user_scaling_factor is between 0 and 1
        if not (0 <= self.user_scaling_factor <= 1):
            raise ValueError("user_scaling_factor must be between 0 and 1.")

        # check if the min_bitrate is less than the max_bitrate
        if self.min_bitrate > self.max_bitrate:
            raise ValueError("min_bitrate cannot be greater than max_bitrate.")

        # check if the min_frame_rate is less than the max_frame_rate
        if self.min_frame_rate > self.max_frame_rate:
            raise ValueError("min_frame_rate cannot be greater than max_frame_rate.")

        # check if the min_structural_accuracy is less than the max_structural_accuracy
        if self.min_structural_accuracy > self.max_structural_accuracy:
            raise ValueError("min_structural_accuracy cannot be greater than max_structural_accuracy.")

        # check if rotation_speed is greater than 0 and less than 1080
        if self.rotation_speed < 360 or self.rotation_speed > 1080:
            raise ValueError("rotation_speed must be greater than 0 and less than 1080.")
        # check if all parameters are positive
        if self.min_bitrate <= 0 or self.max_bitrate <= 0 or self.min_frame_rate <= 0 or self.max_frame_rate <= 0 \
                or self.min_structural_accuracy <= 0 or self.rotation_speed <= 0 or self.quality_weights['ssim'] <= 0 \
                or self.quality_weights['vmaf'] <= 0  \
                or self.user_scaling_factor <= 0 or self.unit_bandwidth_cost <= 0 or self.unit_compute_cost <= 0 \
                or self.min_behavioral_accuracy <= 0 or self.max_behavioral_accuracy > 1:
            raise ValueError("All parameters must be positive.")

        print(f"@VirtualRoom(id={self.id}): All parameters are valid.")

    def calculate_user_scaled_resources(self,
                                        base_resource: float,
                                        num_users: int,
                                        resource_type: str) -> float:
        """Calculate resource needs with user scaling"""
        # req = self.req

        # Get appropriate scaling factor
        if resource_type == 'compute':
            scale_factor = self.user_scale_compute
        else:  # bandwidth
            scale_factor = self.user_scale_bandwidth

        # Calculate user density impact
        density_impact = (num_users / self.max_users) ** self.user_density_factor

        # Calculate shared resource factor
        sharing_factor = 1 - (self.resource_sharing_factor * (1 - density_impact))

        # Scale resources with user count
        scaled_resource = base_resource * (
                pow(num_users, scale_factor) * sharing_factor
        )

        return scaled_resource

    def calculate_ssim(self, user_bitrate):
        """
        This function is used to calculate the SSIM score for the virtual room.
        @param user_bitrate: the bitrate of the user
        @return: the SSIM score
        """
        # v_m is the average rotation speed of VR user m 
        # R_m the required bit rate for the virtual room
        """Calculates SSIM score"""
        b = self.ssim_coeff
        power = -(b['b3'] + b['b4'] * self.rotation_speed)
        ssim = 1 - (b['b1'] + b['b2'] * self.rotation_speed) * (user_bitrate ** power)
        return max(b['b0'], ssim)

    def calculate_vmaf(self, user_bitrate):
        """
        This function is used to calculate the VMAF score for the virtual room.
        @param user_bitrate: the bitrate of the user
        @return: the VMAF score
        """
        # v_m is the average rotation speed of VR user m 
        # R_m the required bit rate for the virtual room
        """Calculates VMAF score"""
        b = self.vmaf_coeff
        vmaf = b['b1'] + b['b2'] * self.rotation_speed + b['b3'] * user_bitrate + b[
            'b4'] * self.rotation_speed * user_bitrate
        return min(100, vmaf)

    def calculate_max_qope(self):
        """
        This function is used to calculate the max QOPE for the virtual room.
        """
        maximum_bitrate_per_room = self.max_bitrate
        ssim = self.calculate_ssim(maximum_bitrate_per_room)
        vmaf = self.calculate_vmaf(maximum_bitrate_per_room)
        return (self.quality_weights['ssim'] * ssim) + (
                    self.quality_weights['vmaf'] * vmaf)  # + (self.BEH_ACC_WEIGHT * 1)

    def calculate_base_compute(self) -> Dict[str, float]:
        """Calculate base compute units needed for VE and DT"""
        # print(type(self.polygon_count), self.polygon_count)
        ve_compute = (
                self.polygon_count / 100_000 +
                self.physics_objects / 100 +
                self.interaction_points / 10
        )

        dt_compute = (
                self.num_sensors / 100 +
                self.state_variables / 100 +
                self.update_frequency / 10
        )
        # print(f've_compute: {ve_compute}, dt_compute: {dt_compute}')

        return {
            've_compute': ve_compute,
            'dt_compute': dt_compute
        }

    def calculate_metrics(self, given_bitrate, num_heads_clients, behavioral_accuracy, given_frame_rate):
        """
        This function is used to calculate the metrics for the virtual room.
        """
        # duration_hours = 1
        duration_hours = 1

        # Calculate SSIM and VMAF scores
        user_ssim = self.calculate_ssim(given_bitrate)  # tested OK
        user_vmaf = self.calculate_vmaf(given_bitrate)  # tested OK

        # Calculate the QOPE score
        raw_visual_quality = (
                self.quality_weights['ssim'] * user_ssim +
                self.quality_weights['vmaf'] * user_vmaf
        )

        # calculate the normalized bitrate and frame rate
        # norm_bitrate = (given_bitrate - self.min_bitrate) / (self.max_bitrate - self.min_bitrate)
        
        # print(f"given_frame_rate: {given_frame_rate}, min_frame_rate: {self.min_frame_rate}, max_frame_rate: {self.max_frame_rate}")
        normalized_responsive_score = (given_frame_rate - self.min_frame_rate) / (
                    self.max_frame_rate - self.min_frame_rate)
        # print(f"normalized_responsive_score: {normalized_responsive_score}")
        error_msg = f"normalized_responsive_score is out of range, {normalized_responsive_score}"
        assert 0.0 <= normalized_responsive_score <= 1.0, error_msg

        # Normalize the QOPE score
        max_qope = self.calculate_max_qope()  # maximum QOPE for the room
        normalized_visual_quality = raw_visual_quality / max_qope
        assert raw_visual_quality <= max_qope, "raw visual quality is greater than max QOPE"

        # Calculate the Immersiveness Score to explicitly include Behavioral Accuracy
        # w1, w2, w3 = 0.4, 0.3, 0.3  # Example weights for bitrate, frame rate, and behavioral accuracy
        immersiveness_score = (
                VISUAL_QUALITY_WEIGHT * normalized_visual_quality +  # Visual quality has highest weight
                RESPONSIVE_WEIGHT * normalized_responsive_score +  # Smooth motion is next important
                BEHAVIORAL_ACCURACY_WEIGHT * behavioral_accuracy  # DT accuracy completes the experience
        )
        # immersiveness_score = round(immersiveness_score, 2)
        #immersiveness_score = immersiveness_score #let's try making it without any rounding.
        assert (behavioral_accuracy >= self.min_behavioral_accuracy) or (behavioral_accuracy <= self.max_behavioral_accuracy), "Behavioral accuracy is out of range"
        assert immersiveness_score <= 1.0 or immersiveness_score >= 0.0, "Immersiveness score is out of range"

        ####################################### Computing units calculation #######################################

        # first calculate the compute needed for VE and DT
        base_compute = self.calculate_base_compute()  # for 1 user and 1 dt.

        # now calculate the compute scaling factors for VE and DT
        compute_scale = pow(given_frame_rate / self.min_frame_rate, 1.1)  # Slightly superlinear
        # the frame rate is for both.
        # print(f"compute_scale because of frame rate: {compute_scale}")

        # Calculate VE computing resources
        ve_compute = self.calculate_user_scaled_resources(
            base_compute['ve_compute'] * compute_scale,
            num_heads_clients,
            'compute'
        ) * duration_hours
        
        
        # Calculate DT computing resources
        dt_compute = self.calculate_user_scaled_resources(
            base_compute['dt_compute'] * behavioral_accuracy * compute_scale,
            num_heads_clients,
            'compute'
        ) * duration_hours

        ####################################### Bandiwdth calculation#######################################
        bandwidth_scale = pow(given_frame_rate / self.min_frame_rate, 0.5)  # Square root relationship

        # Now I need to calculate the bandwidth needed for the ve and the dt
        ve_bandwidth = self.calculate_user_scaled_resources(
            given_bitrate * bandwidth_scale,
            num_heads_clients,
            'bandwidth'
        ) * duration_hours

        dt_bandwidth = self.calculate_user_scaled_resources(
            0.1 * given_bitrate * behavioral_accuracy * bandwidth_scale,
            num_heads_clients,
            'bandwidth'
        ) * duration_hours

        # print(f"ve_compute: {ve_compute}, dt_compute: {dt_compute}")

        ####################################### Cost calculation#######################################

        ve_compute_cost = ve_compute * self.unit_compute_cost
        dt_compute_cost = dt_compute * self.unit_compute_cost

        ve_bandwidth_cost = ve_bandwidth * self.unit_bandwidth_cost
        dt_bandwidth_cost = dt_bandwidth * self.unit_bandwidth_cost

        total_cost = ve_compute_cost + ve_bandwidth_cost + dt_compute_cost + dt_bandwidth_cost

        # print(
        #     f"@{self.__class__.__name__}, Info:\n"
        #     f"number of resources for ve_compute: {ve_compute}\n"
        #     f"number of resources for dt_compute: {dt_compute}\n"
        #     f"number of resources for ve_bandwidth: {ve_bandwidth}\n"
        #     f"number of resources for dt_bandwidth: {dt_bandwidth}\n"
        #     f"ve_compute_cost: {ve_compute_cost}\n"
        #     f"ve_bandwidth_cost: {ve_bandwidth_cost}\n"
        #     f"dt_compute_cost: {dt_compute_cost}\n"
        #     f"dt_bandwidth_cost: {dt_bandwidth_cost}\n"
        #     f"total_cost: {total_cost}"
        # )

        # efficency, cost, computing and bandwidth requirements
        return {
            'immersiveness_score': immersiveness_score,
            'total_cost': total_cost
            # 've_compute': ve_compute,
            # 'dt_compute': dt_compute,
            # 've_bandwidth': ve_bandwidth,
            # 'dt_bandwidth': dt_bandwidth
        }
