"""
This file contains the code for the environment of cooperation.
The environment is a multi-agent environment where each agent is a MSP.
currently, the enviroment will be centralized. then after the proof of concept, we will implement a decentralized environment.
"""
from __future__ import annotations

import csv
import json
import os.path
from collections import OrderedDict
import sys
from typing import List, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType
from gymnasium.spaces import Box

# from HierancyRewarding import HierarchicalRewardCalculator
from helpers.Constants import MIN_IMMERSIVENESS_TO_PASS, PERC_TO_KEEP, MOVING_AVERAGE_WINDOW, HeadType, \
    ACTION_DOABLE_AND_APPLIED
from helpers.GlobalState import GlobalState
from model.Head import Head
from model.MSP import MSP
from model.VirtualRoom import VirtualRoom


def concatenate_dicts(dict1, dict2):
    """
    Concatenate two dictionaries.
    """
    return {**dict1, **dict2}


def normalize_value(value: float, min_value: float, max_value: float) -> float:
    """
    Normalize a value to be between -1 and 1.
    :param value: The value to be normalized.
    :param min_value: The minimum value of the range.
    :param max_value: The maximum value of the range.
    :return: The normalized value between -1 and 1.
    """
    assert min_value <= max_value, "min_value cannot be greater than max_value"
    assert min_value != max_value, "min_value and max_value cannot be the same"
    normalized = 2 * ((value - min_value) / (max_value - min_value)) - 1
    assert -1 <= normalized <= 1, f"Normalized value is out of bounds: min_value: {min_value}, max_value: {max_value}, value: {value} and normalized: {normalized}"
    return normalized


def calc_log(remain, total, steepness=10, shift=0.5):
    steepness = steepness  # Increase steepness for more aggressive scaling
    shift = shift  # Center the sigmoid to make 50% savings more meaningful

    budget_efficiency = remain / total
    budget_reward = 40 / (1 + np.exp(-steepness * (budget_efficiency - shift)))
    return budget_reward


def value_to_discrete(value):
    """
    This function takes a value from -1 to 1 and outputs a discrete number from 0 to 63 inclusive.
    Each discrete value represents an equal-width bin in the continuous space.
    """
    # Ensure the value is within the expected range
    # if value < -2 or value > 2:
    assert -1 <= value <= 1, "Value must be between -1 and 1 inclusive."

    # normalized_value = (value + 2) / 4
    normalized_value = (value + 1) / 2

    # Scale and floor the normalized value to create equal-width bins
    # Multiply by 64 instead of 63 to create 64 bins (0-63)
    # Use min to ensure we don't get 64 as a result for the maximum input value
    discrete_value = min(int(normalized_value * 64), 63)

    return discrete_value


def help_value_to_discrete(value, max_value):
    """
    This function takes a value from -1 to 1 and outputs a discrete number from 0 to 4 inclusive.
    Each discrete value represents an equal-width bin in the continuous space.
    """
    assert -1 <= value <= 1, "Value must be between -1 and 1 inclusive."
    normalized_value = (value + 1) / 2
    discrete_value = min(int(normalized_value * max_value), max_value - 1)
    return discrete_value


class GameCoopEnv(gym.Env):
    """
    This class is the environment for the cooperation game.
    Centralized environment for now.
    """

    ###################### Environment Functions ######################################
    def __init__(self, run_name: str, num_common_heads: int = 0, max_clock: int = 2000, msps_requests: int = 50,
                 extra_info=""):
        # general settings
        self.total_episodical_maximum_so_far = 0
        self.total_number_of_episodes_trained = 0
        self.run_name = run_name
        self.max_clock = max_clock
        self.extra_info = extra_info  # this is a general info about the run.
        # self.my_report_episodes = [100_000,200_000,300_000] #3WILL GET BACK LATER
        self.full_path = f"./results/{self.run_name}/"
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path, exist_ok=True)
            print(f"@{self.__class__.__name__}: Created directory: {self.full_path}")
        # The environment configuration.
        self.vrooms_catalogue = {}
        self.vrooms_list: List[VirtualRoom] = []
        self.msp_list: List[MSP] = []
        self.head_list: List[Head] = []
        self.num_common_heads = num_common_heads  # todo: later on use.
        self.virtual_room_list: List[VirtualRoom] = []
        self.per_msp_reqs = msps_requests  # of requests that needs to be served by each of the msps.
        self.msps_quality_score = 0  # i think its sum of avg imm per msp's heads for each msp.

        # all values in the episode.
        self.episode_timestep_lst: List[int] = []  # to store the timesteps only
        self.episode_need_help_lst: List[int] = []  # to store the dynamic number of msps that needs help.
        self.episode_avg_imrvnss_alive_msp_lst: List[float] = []  # stores the avg immers value divided by alive msps.
        self.episode_overall_avg_imrvnss_lst: List[float] = []  # stores the avg immersiveness value divided by all msps
        self.episode_total_cost_lst: List[float] = []  # stores the episode total cost
        self.episode_avg_cost_per_client: List[float] = []
        self.episode_worked_msps_lst: List[int] = []
        self.episode_avg_cost_alive_lst: List[float] = []
        self.total_timestep_reward: List[float] = []
        self.moving_average_timestep_reward: List[float] = []
        self.last_total_cost: int = 0
        self.last_total_immersiveness: int = 0
        self.last_num_msps_applied: int = 0
        self.last_decoded_actions = 0
        self.last_reward = 0

        # todo: later on use.
        self.avg_total_compute_efficiency: List[float] = []
        self.avg_total_bandwidth_efficiency: List[float] = []

        # todo: later on use.
        assert self.num_common_heads == 0, f"@{self.__class__.__name__}, Error: Common heads are not supported yet"
        assert 0 < self.max_clock <= 5000, f"@{self.__class__.__name__}, Error: max_clock out of range rooms created"
        assert 0 < self.per_msp_reqs <= 1000, f"@{self.__class__.__name__}, Error: msps_requests must be greater than 0 and less than 100"

        # create the virtual rooms
        vroom1 = self.create_virtual_room("LIBRARY")
        vroom2 = self.create_virtual_room("LIBRARY")
        vroom3 = self.create_virtual_room("LIBRARY")
        vroom4 = self.create_virtual_room("LIBRARY")
        vroom5 = self.create_virtual_room("LIBRARY")
        self.vrooms_list.append(vroom1)
        self.vrooms_list.append(vroom2)
        self.vrooms_list.append(vroom3)
        self.vrooms_list.append(vroom4)
        self.vrooms_list.append(vroom5)
        assert len(self.vrooms_list) > 0, f"@{self.__class__.__name__}, Error: No virtual rooms created"

        self.create_heads()  # create the heads
        assert len(self.head_list) > 0, f"@{self.__class__.__name__}, Error: No heads created"

        self.create_msps(msps_requests=self.per_msp_reqs)  # create the msps
        assert len(self.msp_list) > 0, f"@{self.__class__.__name__}, Error: No msps created"
        self.num_msps = len(self.msp_list)

        # create the environment state
        self.state = self.get_state()
        self.observation_space = Box(
            low=np.array([-1.0 for _ in self.state]),
            high=np.array([1.0 for _ in self.state]),
            dtype=np.float64)

        print(f"@{self.__class__.__name__}, Info: Observation space created with shape: {self.observation_space.shape}")
        # create the action space

        # self.action_space_config = [m.get_possible_actions()[1] for m in self.msp_list]
        # print("self.action_space_config=", self.action_space_config)
        # self.action_space_config = [m.get_possible_actions()[1] for _ in range(m.get_num_heads()) for m in
        # self.msp_list] 
        # self.action_space_config = [m.get_possible_actions()[1] for m in self.msp_list]
        self.help_action_space_config = [m.get_possible_help_actions()[1] for m in self.msp_list]
        # print("Sample is ", self.msp_list[0].get_possible_help_actions()[0])
        # create the actions of help to be -1 to 1 
        print(f"@{self.__class__.__name__}, Info: Help action space config: {self.help_action_space_config}")
        # Open to change the action space to discrete 
        # self.action_space = spaces.MultiDiscrete(self.action_space_config)

        # Changed to a continuous action space. without the help action space.
        # self.action_space = Box(
        #     low=np.array([-1.0 for _ in range(self.num_msps)]),
        #     high=np.array([1.0 for _ in range(self.num_msps)]),
        #     dtype=np.float64)

        # Changed to a continuous action space. with the help action space.
        self.action_space = Box(
            low=np.array([-1.0 for _ in range(self.num_msps + self.num_msps)]),
            high=np.array([1.0 for _ in range(self.num_msps + self.num_msps)]),
            dtype=np.float64)

        print(f"@{self.__class__.__name__}, Info: Action space created with shape: {self.action_space.shape}")

    def parse_help(self, partial_action):
        """
        This function is used to parse the help from the action.
        """
        # the input is a list of values of -1 to 1 indicating the help configuration. 
        output = [help_value_to_discrete(val, max_value=m.get_possible_help_actions()[1]) for val, m in
                  zip(partial_action, self.msp_list)]
        return output

    def step(self, action: ActType):
        """
        This function is used to take a step in the environment.
        """
        # in this setting, the action includes the choice of msp(0) ... msp(N)
        # print(f"@{self.__class__.__name__}, Info: Action: {action}")
        # 1. decode the action per each msp & send it to the msp.
        # Open to change the action space if continous 
        action_copy = action[:]  # represents the full action.
        # print(f"@{self.__class__.__name__}, Info: full action: {action_copy}")
        action_msps = [value_to_discrete(act) for act in action_copy[:self.num_msps]]
        # print(f"@{self.__class__.__name__}, Info: action of MSPs to heads: {action_msps}")
        help_action = self.parse_help(action_copy[self.num_msps:])
        # print(f"@{self.__class__.__name__}, Info: help_action_parsed: {help_action}")
        output_dict = self.apply_help(help_action)
        # print in green color
        # print(f"\033[92m@{self.__class__.__name__}, Info: output_dict: {output_dict}\033[0m")

        decoded_actions = self.parse_action(action_msps, operation_mode=0)
        # print(f"@{self.__class__.__name__}, Info: Decoded actions: {decoded_actions}")
        res = self.apply_decoded_action(decoded_actions, operation_mode=0)
        # print in pink color
        # print(f"\033[95m@{self.__class__.__name__}, Info: res: {json.dumps(res, indent=4)}\033[0m")

        # recording some of the stats.
        total_cost = res["total_cost"]  # total cost of doing that action.
        #print(f"@{self.__class__.__name__}, Info: Total cost: {total_cost}")
        total_imm = res['total_imm']  # Total immersion of that action.
        num_msps_applied = res['num_msps_applied']  # Number of msps applied.
        # sat_pen = res['sat_pen'] # satisfaction & penality.
        self.last_decoded_actions = decoded_actions
        self.last_total_cost = total_cost
        self.last_total_immersiveness = total_imm
        self.last_num_msps_applied = num_msps_applied
        # print(json.dumps(sat_pen, indent=4))
        # reward, moving_avg = self.calculate_reward_new_intermediate2(sat_pen)  # worked good.

        # Calculating the rewards.
        reward, moving_avg = self.calc_inter_multi(res,output_dict)
        # reward, moving_avg = 0,0

        # the following 4 lines were used with the intermediate reward 2
        self.total_timestep_reward.append(reward)

        # Increment clock.
        GlobalState.increment_clock()

        # check if done
        done, reason_string, termination_reason = self.is_done()

        # check next state.
        self.state = self.get_state()

        # update the number of steps for each msp.
        for m in self.msp_list:
            if not m.is_msp_finished_budget()[0]:
                m.step_counter += 1

        if done:
            msps_budgets = {f"msp_{m.id}_budget_left": m.get_budget() for m in self.msp_list}
            q = []
            for msp in self.msp_list:
                avg_imm_heads_of_msp = []
                msp_s = 0
                b = 0
                imm = 0
                for hhead in msp.heads:
                    msp_s += (hhead.number_of_done_requests * hhead.moving_average_immersion.get_average()) / (
                            msp.num_requests * hhead.get_target_immersiveness())
                    # msp_s += (hhead.moving_average_immersiveness.get_average()/hhead.get_target_immersiveness())
                    b_noclip = ((
                                        sum(hhead.moving_average_immersion.get_vals_list()) / msp.num_requests) / hhead.get_target_immersiveness())

                    # imm_noclip = (sum(hhead.moving_average_immersion.get_vals_list()) / msp.num_requests)
                    # imm_noclip = (sum(hhead.moving_average_immersion.get_vals_list()) / msp.num_requests_done)
                    imm_noclip = (hhead.moving_average_immersion.get_sum() / msp.num_requests_done)
                    imm_clipped = np.clip(imm_noclip, 0, 1)
                    imm += imm_clipped
                    msp.lst_final_imm.append(imm_clipped)
                    b += np.clip(round(b_noclip, 2), 0, 1)  # this avoids over-accomplish (keeps between 0 and 1 )
                    avg_imm_heads_of_msp.extend(hhead.moving_average_immersion.get_vals_list())
                q.append(np.clip(msp_s / len(msp.heads), 0, 1))
                inter_avg = np.sum(avg_imm_heads_of_msp) / (msp.num_requests * len(msp.heads))
                msp.final_reward = inter_avg
                msp.b = b  # this is the sum of b for all heads of the msp
                msp.b_avg = b / len(msp.heads)  # this is the average of b for all heads of the msp
                msp.achvd_immers = imm / len(msp.heads)
                self.total_number_of_episodes_trained += 1
                # if self.total_number_of_episodes_trained%500 == 0:
                # print("msp.id", msp.id, "msp.achvd_immers: ", msp.achvd_immers, " msp.b_avg(85%): ", msp.b_avg,
                #       " Steps:", msp.num_requests_done)
                # print("msp.achvd_immers: ", msp.achvd_immers, " msp.b_avg(85%): ", msp.b_avg)
            qq = {f"msp_q_{_id}": val for _id, val in enumerate(q)}
            self.msps_quality_score = sum(val for val in qq.values())

            # Terminal reward.
            reward = self.calc_term_multi(termination_reason)
            self.total_timestep_reward.append(reward)

            # Write the following to the file.
            total_budget = sum(m.initial_budget for m in self.msp_list)
            remaining_budget = sum(m.get_budget() for m in self.msp_list)
            reqs_fullfilled = sum(m.num_requests_fullfilled for m in self.msp_list)
            reqs_done= [m.num_requests_done for m in self.msp_list]
            row_data = OrderedDict({
                # "num_steps": GlobalState.get_clock() - 1,
                "max_msp_reqs": np.max(reqs_done),
                "min_msp_reqs": np.min(reqs_done),
                "tot_req_done": np.sum(reqs_done),
                "num_requests_fulfilled": reqs_fullfilled,
                "total_satisfied_requests_percentage": reqs_fullfilled / np.sum(reqs_done)* 100,
                # "last_num_msps_applied": self.last_num_msps_applied,
                "episode_val": reqs_fullfilled , # need to multiply by immersion. 
                "total_reward": sum(self.total_timestep_reward),
                # "moving_avg_reward": sum(self.moving_average_timestep_reward),
                # "average_reward_per_steps": sum(self.total_timestep_reward) / GlobalState.get_clock(),
                "total_cost": sum(self.episode_total_cost_lst),
                "avg_cost_per_client": sum(self.episode_avg_cost_per_client),
                "avg_cost_alive": sum(self.episode_avg_cost_alive_lst),
                # "total_imm": sum(self.episode_overall_avg_imrvnss_lst),
                "avg_head_imrvnss_alive": np.mean(self.episode_avg_imrvnss_alive_msp_lst),
                "avg_head_imrvnss_overall": np.mean(self.episode_overall_avg_imrvnss_lst),
                # "need_help": sum(self.episode_need_help_lst),
                "worked_msps": sum(self.episode_worked_msps_lst),
                "avg_worked_msps": np.mean(self.episode_worked_msps_lst),
                # "help_budget_invested": 0,
                # "total_budget": total_budget,
                # "remaining_budget": remaining_budget,
                # "consumed_budget": total_budget - remaining_budget,
                "total_helped_times": sum(m.total_helped_times for m in self.msp_list),  # can put it per msp if needed.
                "total_help_received": sum(m.total_help_received for m in self.msp_list),
                # can put it per msp if needed.
                # "runname": self.run_name,
            })
            # row_data = concatenate_dicts(row_data, msps_budgets)
            # row_data = concatenate_dicts(row_data, qq)
            if self.extra_info != "":
                row_data["extra_info"] = self.extra_info
            self.write_to_file(row_data, f"summary.csv")
            # print(f"@{self.__class__.__name__}, Info: Episode finished with reason: {reason}")

        return self.state, reward, done, False, {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        # todo: complete
        GlobalState.reset_clock()
        # resetting historical values
        self.episode_timestep_lst.clear()  # to store the timesteps only
        self.episode_need_help_lst.clear()  # to store the dynamic number of msps that needs help.
        self.episode_avg_imrvnss_alive_msp_lst.clear()  # stores the avg immers value divided by alive msps.
        self.episode_overall_avg_imrvnss_lst.clear()  # stores the avg immersiveness value divided by all msps
        self.episode_total_cost_lst.clear()  # stores the episode total cost
        self.episode_avg_cost_per_client.clear()
        self.episode_worked_msps_lst.clear()
        self.episode_avg_cost_alive_lst.clear()
        self.total_timestep_reward.clear()
        self.moving_average_timestep_reward.clear()
        self.last_total_cost = 0
        self.last_total_immersiveness = 0
        self.last_num_msps_applied = 0
        self.last_decoded_actions = 0
        self.last_reward = 0
        self.avg_total_compute_efficiency.clear()
        self.avg_total_bandwidth_efficiency.clear()
        # resetting the heads, msps
        for m in self.msp_list:
            m.episode_reset()
        for h in self.head_list:
            h.episode_reset()

        # should go to 10 like this.
        # if len(self.my_report_episodes) > 0:
        #     if self.total_number_of_episodes_trained == self.my_report_episodes[0]:
        #         self.my_report_episodes.pop(0)
        #         for h in self.head_list:
        #             h.increase_num_users(1)
        #         print("increased number of users for all heads")

        self.state = self.get_state()
        return self.state, {"data": 0}

    def render(self):
        """
        Renders a visualization of MSPs with their budgets and metrics using progress bars
        """
        import matplotlib.pyplot as plt
        # print(matplotlib.get_backend())

        # Create figure and axis
        plt.clf()
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(10, 14))
        fig.suptitle(f'MSP Status Dashboard\nLast Actions: {self.last_decoded_actions} Reward: {self.last_reward}')

        # Plot overall metrics
        metrics_names = ['Total Cost', 'Total Immersiveness', 'Active MSPs']
        metrics_values = [self.last_total_cost, self.last_total_immersiveness, self.last_num_msps_applied]
        colors = ['gold', 'mediumorchid', 'teal']

        ax0.bar(metrics_names, metrics_values, color=colors)
        ax0.set_title('Overall System Metrics')
        ax0.tick_params(axis='x', rotation=30)
        # Add values as text on top of bars
        for i, v in enumerate(metrics_values):
            ax0.text(i, v, f'{v:.2f}', ha='center', va='bottom')

        # Plot budget bars
        msp_names = [f'MSP {msp.id}' for msp in self.msp_list]
        budget_percentages = [msp.get_budget() / msp.initial_budget for msp in self.msp_list]

        # Budget bars
        ax1.barh(msp_names, budget_percentages, color='skyblue')
        ax1.set_title('MSP Budget Status')
        ax1.set_xlim(0, 1)
        ax1.set_xlabel('Budget Remaining (%)')

        # Add budget values as text
        for i, v in enumerate(budget_percentages):
            ax1.text(v, i, f'${self.msp_list[i].budget:.2f}', va='center')

        # Plot metrics
        metrics = []
        for msp in self.msp_list:
            if len(msp.heads) > 0 and len(msp.heads[0].hist) > 0 and msp.is_msp_finished_budget()[0] == False:
                last_metrics = list(msp.heads[0].hist.values())[-1]
                metrics.append({
                    'cost': last_metrics['total_cost'],
                    'immersiveness': last_metrics['immersiveness_score']
                })
            else:
                metrics.append({'cost': 0, 'immersiveness': 0})

        costs = [m['cost'] for m in metrics]
        immersiveness = [m['immersiveness'] for m in metrics]

        # Plot costs
        ax2.bar(msp_names, costs, color='lightcoral')
        ax2.set_title('MSP Costs')
        ax2.tick_params(axis='x', rotation=45)
        # Add cost values above bars
        for i, cost in enumerate(costs):
            ax2.text(i, cost, f'${cost:.2f}', ha='center', va='bottom')

        # Plot immersiveness
        ax3.bar(msp_names, immersiveness, color='lightgreen')
        ax3.set_title('MSP Immersiveness Scores')
        ax3.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.pause(0.7)
        # plt.show()

    ###################### Helper Functions ######################################

    def create_virtual_room(self, room_type: str):
        """
        Create a new VirtualRoom of the specified type.
        """
        assert len(self.vrooms_catalogue) == 0, f"@{self.__class__.__name__}, Error: Virtual rooms already have content"
        assert room_type.upper() in ["LIBRARY", "ARENA",
                                     "GALLERY"], f"@{self.__class__.__name__}, Error: Invalid room type: {room_type}"
        if room_type.upper() == "LIBRARY":
            return VirtualRoom(
                min_bitrate=20,
                max_bitrate=25,
                min_frame_rate=30,
                max_frame_rate=60,
                min_structural_accuracy=0.6,
                max_structural_accuracy=1,
                rotation_speed=400,
                quality_weights={'ssim': 0.5, 'vmaf': 0.5},
                user_scaling_factor=0.7,
                unit_bandwidth_cost=0.01,
                unit_compute_cost=0.01,
                min_behavioral_accuracy=0.5,
                max_behavioral_accuracy=1,
                max_users=10,
                user_scale_compute=0.7,
                user_scale_bandwidth=0.9,
                user_density_factor=0.8,
                resource_sharing_factor=0.6,
                polygon_count=200_000,
                physics_objects=50,
                interaction_points=5,
                num_sensors=50,
                state_variables=30,
                update_frequency=2
            )
        elif room_type.upper() == "ARENA":
            return VirtualRoom(
                min_bitrate=30,
                max_bitrate=50,
                min_frame_rate=60,
                max_frame_rate=120,
                min_structural_accuracy=0.5,
                max_structural_accuracy=1,
                rotation_speed=720,
                quality_weights={'ssim': 0.3, 'vmaf': 0.7},
                user_scaling_factor=0.5,
                unit_bandwidth_cost=0.01,
                unit_compute_cost=0.01,
                min_behavioral_accuracy=0.5,
                max_behavioral_accuracy=1,
                max_users=100,
                user_scale_compute=0.85,
                user_scale_bandwidth=0.85,
                user_density_factor=0.8,
                resource_sharing_factor=0.5,
                polygon_count=500_000,
                physics_objects=200,
                interaction_points=30,
                num_sensors=200,
                state_variables=100,
                update_frequency=10
            )
        elif room_type.upper() == "GALLERY":
            return VirtualRoom(
                min_bitrate=25,
                max_bitrate=35,
                min_frame_rate=30,
                max_frame_rate=60,
                min_structural_accuracy=0.8,
                max_structural_accuracy=1.0,
                rotation_speed=400,
                quality_weights={'ssim': 0.7, 'vmaf': 0.3},
                user_scaling_factor=0.6,
                user_scale_compute=0.7,
                user_scale_bandwidth=0.8,
                unit_bandwidth_cost=0.01,
                unit_compute_cost=0.01,
                user_density_factor=0.7,
                resource_sharing_factor=0.8,
                polygon_count=300_000,
                physics_objects=20,
                interaction_points=15,
                num_sensors=30,
                state_variables=20,
                update_frequency=1
            )
        else:
            raise ValueError(f"Unknown room type: {room_type}")

    def create_heads(self):

        # to make the limited budget scenario. 
        # head1 = Head(num_users=5, room=self.vrooms_list[0], htype=HeadType.LOCAL, target_immersiveness=0.85)
        # head2 = Head(num_users=7, room=self.vrooms_list[1], htype=HeadType.LOCAL, target_immersiveness=0.85)
        # head3 = Head(num_users=9, room=self.vrooms_list[2], htype=HeadType.LOCAL, target_immersiveness=0.85)
        # head4 = Head(num_users=11, room=self.vrooms_list[3], htype=HeadType.LOCAL, target_immersiveness=0.85)
        # head5 = Head(num_users=13, room=self.vrooms_list[4], htype=HeadType.LOCAL, target_immersiveness=0.85)

        # head1 = Head(num_users=5, room=self.vrooms_list[0], htype=HeadType.LOCAL, target_immersiveness=0.85)
        # head2 = Head(num_users=5, room=self.vrooms_list[1], htype=HeadType.LOCAL, target_immersiveness=0.85)
        # head3 = Head(num_users=5, room=self.vrooms_list[2], htype=HeadType.LOCAL, target_immersiveness=0.85)
        # head4 = Head(num_users=5, room=self.vrooms_list[3], htype=HeadType.LOCAL, target_immersiveness=0.85)
        # head5 = Head(num_users=5, room=self.vrooms_list[4], htype=HeadType.LOCAL, target_immersiveness=0.85) 

        head1 = Head(num_users=5, room=self.vrooms_list[0], htype=HeadType.LOCAL, target_immersiveness=MIN_IMMERSIVENESS_TO_PASS)
        head2 = Head(num_users=5, room=self.vrooms_list[1], htype=HeadType.LOCAL, target_immersiveness=MIN_IMMERSIVENESS_TO_PASS)
        # head3 = Head(num_users=5, room=self.vrooms_list[2], htype=HeadType.LOCAL, target_immersiveness=0.85)
        # head4 = Head(num_users=5, room=self.vrooms_list[3], htype=HeadType.LOCAL, target_immersiveness=0.85)
        # head5 = Head(num_users=5, room=self.vrooms_list[4], htype=HeadType.LOCAL, target_immersiveness=0.85)

        self.head_list.append(head1)
        self.head_list.append(head2)
        # self.head_list.append(head3)
        # self.head_list.append(head4)
        # self.head_list.append(head5)
        for h in self.head_list:
            print(f"@{self.__class__.__name__}, Info: Created head {h.id}, has {h.num_users} users")
        # print(
        #     f"@{self.__class__.__name__}, Info: Created 2 heads, first has {head1.num_users} users and second has {head2.num_users} users")

    def create_msps(self, msps_requests: int = 50):
        # tot_bud = 700 / 2
        # t = tot_bud / 2
        # total_budget_per_msp = 37/3
        # total_budget_per_msp = 29/3 # maximum config cost  of 63,63,63 

        # setting used in limited budget case.
        # total_budget_per_msp = (30/3) *0.80 #0.45

        msp1 = MSP(heads=[self.head_list[0]], budget=100, num_requests=msps_requests)
        msp2 = MSP(heads=[self.head_list[1]], budget=50, num_requests=msps_requests)

        # msp3 = MSP(heads=[self.head_list[2]], budget=total_budget_per_msp, num_requests=msps_requests,
        #            heads_target_imm=0.85)
        # msp4 = MSP(heads=[self.head_list[3]], budget=total_budget_per_msp, num_requests=msps_requests, heads_target_imm=0.85)
        # msp5 = MSP(heads=[self.head_list[4]], budget=total_budget_per_msp, num_requests=msps_requests, heads_target_imm=0.85)
        self.msp_list.append(msp1)
        self.msp_list.append(msp2)

        msp1.add_neighbor(msp2)
        msp2.add_neighbor(msp1)

        # self.msp_list.append(msp3)
        # self.msp_list.append(msp4)
        # self.msp_list.append(msp5)
        for msp in self.msp_list:
            print(
                f"@{self.__class__.__name__}, Info: Created msp {msp.id}, has {len(msp.heads)} heads and budget={msp.budget}")

    def is_done(self):
        """
        This function is used to check if the environment is finished.
        """
        # cond1 = False 
        # cond2 = False 
        # cond3 = False 
        # 1. All msps budget is below threshold.

        count = sum(1 for m in self.msp_list if m.is_msp_finished_budget()[0])
        if count == len(self.msp_list):
            # cond1 = True 
            return True, "All msps budget is below threshold", "BUDG_FINISHED"

        # 2. Number of requests passed or reached the maximum number of requests.
        total_requests = sum(m.num_requests for m in self.msp_list)
        total_fulfilled_requests = sum(m.num_requests_fullfilled for m in self.msp_list)

        # # added new.
        # count = sum(1 for m in self.msp_list if m.is_msp_finished_budget()[0])
        # if count > 0 and total_fulfilled_requests < total_requests :
        #     return True, "One msps budget is below threshold", "BUDG_FINISHED2"

        # the following 2 conditions has been removed.
        # print(f"@{self.__class__.__name__}, Info: Total requests: {total_requests}, Total fulfilled requests: {total_fulfilled_requests}")
        # if total_fulfilled_requests >= total_requests:
        #     return True, "All requests are fulfilled", "GOAL_REACHED"

        # 3. Time is up.
        # if GlobalState.get_clock() >= max(msp.num_requests for msp in self.msp_list):
        # cond2 = True
        #     return True, "Time is up", "TIME_UP"

        return False, "Not finished", "NOT_FINISHED"

    def get_state(self):
        """
        This function is used to get the state of the environment.
        the state include: 
        a) The Budget Left For each msp
        b) The Total Number Of Users the msp handles ( summation of all users in all heads)
        c) System clock relative to the maximum clock i would set (400). 
        d) help requests 
        """
        s = []
        # This was working previously. for comparison or whatever please change it back.
        # budgets_left = [normalize_value(m.get_budget(), 0, m.initial_budget) for m in self.msp_list] # absolute value.

        budgets_left = [normalize_value(m.get_budget_percentage(), 0, 1) for m in self.msp_list]
        s.extend(budgets_left)  # budget left for each msp.
        s.extend([normalize_value(m.is_msp_finished_budget()[0], 0, 1) for m in self.msp_list])

        # the total users is normalized betwen 0 and the maximum number of users that can be handled by virtual rooms
        # hosted by the heads which the msp serves.
        system_clock = GlobalState.get_clock()

        # todo: open in the second phase.
        # help_requests = [self.normalize_value(m.need_help(), 0, 1) for m in self.msp_list]
        # time_roll = self.normalize_value(system_clock, 0, self.max_clock)

        # time_roll = normalize_value(system_clock, 0, max(msp.num_requests for msp in self.msp_list))
        # s.append(time_roll)  #clock

        total_users = [normalize_value(m.get_total_num_clients(), 0, sum(head.room.max_users for head in m.heads))
                       for m in self.msp_list]
        # msps_finished = [normalize_value(m.is_msp_finished_budget()[0], 0, 1) for m in self.msp_list]
        # msps_requests_fulfilled = [normalize_value(m.num_requests_fullfilled / m.num_requests, 0, 1) for m in
        #                            self.msp_list]

        # s.extend(total_users) # total number of users each msp serves. (used for scaling)


        # s.extend(msps_requests_fulfilled)  # the percentage of requests that are fulfilled.
        return np.array(s)

    def parse_action(self, action: ActType, operation_mode: int = 0):
        """
        This function is used to parse and set the action for each msp.
        The operation mode specifies which action structure is used so that we can decode it.
        """
        if operation_mode == 0:
            # each msp has 1 action only. [64,4096...etc]
            assert type(self.msp_list[
                            0].actions_dict) == dict, f"@{self.__class__.__name__}, Error: Actions dict is not a " \
                                                      f"dictionary"
            decoded_actions = [self.msp_list[i].actions_dict.get(action[i], "Invalid Action") for i in
                               range(len(self.msp_list))]
            # print(f"@{self.__class__.__name__}, Info: Decoded actions: {decoded_actions}")
            return decoded_actions
        elif operation_mode == 1:
            # each msp has more than one action according to how many heads it has. 
            pass
        else:
            raise ValueError(f"Unknown operation mode: {operation_mode}")

    def apply_decoded_action(self, decoded_actions, operation_mode=0):
        """
        One of the important functions in the environment.
        it does the following:
        1. Takes the decoded actions of the msps, then applies them to the different msps & heads. 
        2. Calculates the total cost, total immersiveness, and the number of msps that applied their actions.
        3. calculates some of the metrics for the episode. 
        """
        res = {}
        if operation_mode == 0:
            # Loop over the msps and apply the action to their heads.
            satisfiaction_penalities_dict = {}
            _timestep_total_cost = 0
            _timestep_total_immersiveness = 0
            _timestep_cost_avg_per_users = 0
            _timestep_num_msps_applied = 0
            _timestep_all_heads = 0  # all the number of heads.
            _timestep_alive_heads = 0  # corresponding alive number of heads with respect to msps.
            # loop over msps. apply the action to each msp.
            for i in range(len(self.msp_list)):
                # Check if the action is doable by the msp or not.
                # todo: before the check apply add the help received to the msp.
                res = self.msp_list[i].check_apply_msp_action(decoded_actions[i])
                act_applied_flag = res["action_flag"]
                total_cost = res["total_cost"]
                total_immersiveness = res["total_immersiveness"]
                num_satisfied_requests = res["num_satisfied_requests"]
                penalities = res["penalties"]
                penalty_satisfaction_list = res["penalty_satisfaction_list"]
                # print("From envCooperation, penalities",penalities)

                _timestep_all_heads += self.msp_list[i].get_num_heads()
                if act_applied_flag == ACTION_DOABLE_AND_APPLIED:
                    _timestep_num_msps_applied += 1
                    self.msp_list[i].num_requests_done += 1
                    _timestep_alive_heads += self.msp_list[i].get_num_heads()
                    no_clients_msp = self.msp_list[i].get_total_num_clients()
                    _timestep_total_cost += total_cost
                    _timestep_total_immersiveness += total_immersiveness
                    _timestep_cost_avg_per_users += total_cost / no_clients_msp
                    satisfiaction_penalities_dict[self.msp_list[i].id] = {
                        "msp_id": self.msp_list[i].id,
                        "applied": True,
                        "penalities": penalities,
                        "num_satisfied_requests": num_satisfied_requests,
                        "penalty_satisfaction_list": penalty_satisfaction_list,
                        "number_of_heads": len(self.msp_list[i].heads)
                    }
                else:
                    satisfiaction_penalities_dict[self.msp_list[i].id] = {
                        "msp_id": self.msp_list[i].id,
                        "applied": False,
                        "penalities": 0,
                        "num_satisfied_requests": 0,
                        "penalty_satisfaction_list": [],
                        "number_of_heads": len(self.msp_list[i].heads)
                    }

                # else:
                #     print(
                #         f"@{self.__class__.__name__}, Warning: Action not applied for msp {i}, total_cost: {total_cost}, total_immersiveness: {total_immersiveness}")
            self.episode_timestep_lst.append(GlobalState.get_clock())  # adding current clock
            self.episode_total_cost_lst.append(_timestep_total_cost)  # adding total cost at this clock.
            if _timestep_alive_heads == 0:
                _timestep_alive_heads = 0.0001
            self.episode_avg_cost_alive_lst.append(_timestep_total_cost / _timestep_alive_heads)
            self.episode_worked_msps_lst.append(_timestep_num_msps_applied)  # adding how many active msps now.
            self.episode_overall_avg_imrvnss_lst.append(_timestep_total_immersiveness / _timestep_all_heads)
            self.episode_avg_imrvnss_alive_msp_lst.append(_timestep_total_immersiveness / _timestep_alive_heads)
            self.episode_avg_cost_per_client.append(_timestep_cost_avg_per_users)
            res["total_cost"] = _timestep_total_cost
            res['total_imm'] = _timestep_total_immersiveness
            res['num_msps_applied'] = _timestep_num_msps_applied
            res['sat_pen'] = satisfiaction_penalities_dict
            return res
            # return _timestep_total_cost, _timestep_total_immersiveness, _timestep_num_msps_applied, satisfiaction_penalities_dict
        else:
            raise ValueError(f"Unknown operation mode: {operation_mode}")

    def apply_help(self, help_action):
        """
        This function is used to apply the help to the neighbors of the msps.
        the help action is a list of numbers, each number represents the configuration index. 
        """
        output_dict = {msp.id: {"Helped": False, "Helped_list": []} for msp in self.msp_list}
        
        # First pass: identify MSPs that need help
        msps_needing_help = [msp for msp in self.msp_list if msp.is_msp_finished_budget()[0]]

        # loops over all msps. 
        for msp, help_val in zip(self.msp_list, help_action):
            # need to decode the help action for this specfic msp 
            decoded_help_ = msp.get_possible_help_actions()[0].get(help_val)
            # print("decoded helps:",decoded_help_)
            # print(f"@{self.__class__.__name__}, Info: MSP {msp.id} decoded_help_: {decoded_help_}")

            if msp in msps_needing_help:
                # skip msps that need help themselves they cannot help others.
                continue
            
            if all(x == 0 for x in decoded_help_):
                # the msp did not help any of the neighoprs
                # print("no help from msp.id",msp.id)
                continue
            else:
                total_cost_of_help = 0
                # loops over all neighbors of the msp.
                for neighbor, help_percentage in zip(msp.neighbors, decoded_help_):
                    # nei_tot_cost_100 = neighbor.get_cost_per_action().get(1) # represents the cost of high immerersion action 
                    # total_cost_of_help += nei_tot_cost_100 * help_percentage
                    budget_percentage = msp.get_budget() * help_percentage
                    total_cost_of_help += budget_percentage
                # after collecting the cost of the help, we need to assign this help to the neighbor (only if budget allows.)

                # check if the budget allows the help.
                msp_budget = msp.get_budget()
                msp_initial_budget = msp.initial_budget
                # checks if we have enough budget to help.

                cond1 = msp_budget > total_cost_of_help
                # checks if we have enough budget to keep after applying the help.
                # cond2 = (msp_budget - total_cost_of_help) > (msp_initial_budget * PERC_TO_KEEP) 
                cond2 = True
                if cond1 and cond2:
                    # print(f"\033[94m@{self.__class__.__name__}, Info: cond1: {cond1}, cond2: {cond2}\033[0m")
                    # apply the help.
                    # loops over all neighbors of the msp.
                    for neighbor, help_percentage in zip(msp.neighbors, decoded_help_):
                        # nei_tot_cost_100 = neighbor.get_cost_per_action().get(1)
                        # nei_tot_cost_100 = neighbor.get_cost_per_action().get(1)
                        help_cost = msp.get_budget() * help_percentage
                        # increase the budget of neighbor 
                        cond3 = True  # checks if the neighbor needs help or not.
                        # cond3 = neighbor.is_msp_finished_budget()[0]  # checks if the neighbor needs help or not.
                        # cond3 = False # checks if the neighbor needs help or not.
                        # print(f"\033[94m@{self.__class__.__name__}, Info: cond3: {cond3}\033[0m")
                        if cond3:
                            # amount = help_cost
                            neighbor.increase_budget(help_cost)
                            msp.decrease_budget(help_cost)
                            output_dict[msp.id]["Helped_list"].append(neighbor.id)
                            output_dict[msp.id]["Helped"] = True
                            msp.total_helped_times += 1 
                            neighbor.total_help_received += help_cost
                            # print in red color.
                            print(f"\033[91m@{self.__class__.__name__}, Info: MSP:{msp.id} helped MSP{neighbor.id} with {help_cost}, about {help_percentage*100}% of its remaining budget and {round(help_cost/msp.initial_budget*100,4)}% of its initial budget \033[0m")

        return output_dict

    def calculate_reward_new(self, satisfiaction_penalities_dict):
        """
        Reward based on request satisfaction progress
        """
        total_satisfied_this_step = 0
        total_heads = 0
        satisfaction_progress = 0
        

        for key, vals in satisfiaction_penalities_dict.items():
            if vals["applied"]:
                total_heads += vals["number_of_heads"]
                total_satisfied_this_step += vals["num_satisfied_requests"]

                # Calculate satisfaction progress for each MSP
                for target_imm, current_imm in [(rec["details"]["target_immersiveness"],
                                                 rec["details"]["current_immersiveness"])
                                                for rec in vals["penalty_satisfaction_list"]]:
                    satisfaction_ratio = np.clip(current_imm / target_imm, 0, 1)
                    satisfaction_progress += satisfaction_ratio

        # Average satisfaction progress across all heads
        avg_satisfaction = satisfaction_progress / total_heads if total_heads > 0 else 0

        # Reward for this timestep
        reward = (total_satisfied_this_step / total_heads if total_heads > 0 else -1)  # + 0.5 * avg_satisfaction
        reward = np.clip(reward, -1, 1)

        return reward, self.calculate_moving_average(reward)

    def calculate_moving_average(self, reward):
        # Calculate the moving average of the reward over the last 100 steps
        self.moving_average_timestep_reward.append(reward)
        if len(self.moving_average_timestep_reward) > MOVING_AVERAGE_WINDOW:
            self.moving_average_timestep_reward.pop(0)
        return np.mean(self.moving_average_timestep_reward)

    def write_to_file(self, row_data: OrderedDict, file_name: str):
        """
        Writes a row of data to the summary CSV file.
        @param row_data: OrderedDict where keys are column names and values are the data to write
        """
        assert file_name.endswith(".csv"), f"@{self.__class__.__name__}, Error: File name must end with .csv"
        if not os.path.exists(os.path.join(self.full_path, file_name)):
            header = row_data.keys()
            with open(os.path.join(self.full_path, file_name), "w") as f:
                writer = csv.writer(f)
                writer.writerow(header)
        with open(os.path.join(self.full_path, file_name), "a") as f:
            writer = csv.writer(f)
            writer.writerow(row_data.values())

    def calc_inter_multi(self, param, output_dict):
        """
        Intermediate reward function focused on maximizing requests above minimum immersion.
        """
        satisfiaction_penalities_dict = param["sat_pen"]
        imms = []
        total_heads = 0

        # Collect immersion values for all heads
        for _, vals in satisfiaction_penalities_dict.items():
            if vals["applied"]:
                total_heads += vals["number_of_heads"]
                for h in vals["penalty_satisfaction_list"]:
                    imms.append(h["details"]["current_immersiveness"])
            else:
                # Penalize inactive heads
                imms.extend([0] * vals["number_of_heads"])

        if not imms:
            return -1.0, -1.0  # Strong penalty if no immersion values

        reward = 0.0
        
        # Calculate success with efficiency incentive
        efficient_successes = 0
        excessive_immersion = 0
        failures = 0
        
        for imm in imms:
            if imm >= MIN_IMMERSIVENESS_TO_PASS: # greater than the minimum immersiveness.
                # Check if immersion is efficiently close to target
                if imm <= MIN_IMMERSIVENESS_TO_PASS * 1.1:  # Within 10% of minimum
                    efficient_successes += 1.5
                else:
                    # Success but with diminishing returns for excessive immersion
                    efficiency_factor = 0.5 - min(0.3, (imm - MIN_IMMERSIVENESS_TO_PASS) / MIN_IMMERSIVENESS_TO_PASS)
                    efficient_successes += efficiency_factor
                    excessive_immersion += 1
            else:# less than the minimum immersiveness.
                failures += 1
        
        # Reward successful requests but with efficiency incentive
        reward += (efficient_successes * 0.15)
        # help_actions = sum(1 for msp in self.msp_list if len(output_dict[msp.id]["Helped_list"]) > 0)
        # reward += help_actions * 0.2
        
        # Small penalty for excessive immersion to discourage wasting resources
        reward -= (excessive_immersion * 0.01)
        
        # Larger penalty for failing to meet minimum immersion
        reward -= (failures * 0.2)

        # Normalize final reward to [-1, 1] range
        # reward = np.clip(reward, -1.0, 1.0)

        return reward, self.calculate_moving_average(reward)

    def calc_term_multi(self, termination_reason: str):
        """
        Terminal reward function that focuses on maximizing applied requests while maintaining
        minimum immersiveness standards.
        """

        # reward = 0
        # # Calculate the average number of requests done by all msps.
        # # total_requests_done = sum(m.num_requests_done for m in self.msp_list)
        # # minimum_requests_done = min(m.num_requests_done for m in self.msp_list)
        # reqs_fullfilled = [m.num_requests_fullfilled for m in self.msp_list]
        # minimum_requests_done = min(reqs_fullfilled)

        # # calculate the average immersion on every requests done (value of immersion on every head)
        # # summ_imm_across_heads = 0
        # # number_of_global_req = 0  # number of heads requests
        # all_imm  = [imm for msp in self.msp_list for imm in msp.lst_final_imm]
        # avg_imm_msps = sum(all_imm) / len(all_imm)
        # # avg_imm_msps = sum(msp.achvd_immers for msp in self.msp_list) / self.num_msps
        
        # # reward = minimum_requests_done * np.clip(avg_imm_msps/MIN_IMMERSIVENESS_TO_PASS, 0, 1)
        # # print the following on one line in green color.
        # Focus on total requests fulfilled across MSPs
        reqs_fullfilled = [m.num_requests_fullfilled for m in self.msp_list]
        total_requests_fulfilled = sum(reqs_fullfilled)
        minimum_requests_done = min(reqs_fullfilled)
        
        # Calculate average immersion
        all_imm = [imm for msp in self.msp_list for imm in msp.lst_final_imm]
        avg_imm_msps = sum(all_imm) / len(all_imm) if all_imm else 0
        
        # New: Calculate balance factor to reward more evenly distributed success
        # balance_factor = 0
        # if len(reqs_fullfilled) > 1:
        #     # Lower variance (more balanced) is better
        #     variance = np.var(reqs_fullfilled)
        #     balance_factor = np.exp(-variance/5)  # Exponential decay for variance
        
        # Total fulfilled requests are now the primary metric
        reward = total_requests_fulfilled
        
        # Bonus for minimum requests being high
        reward += minimum_requests_done * 0.5
        
        # Bonus for helping behavior
        total_helped_times = sum(m.total_helped_times for m in self.msp_list)
        reward += total_helped_times * 0.3
        
        # Bonus for balanced performance
        # reward += balance_factor * 2.0
        
        # Apply immersion multiplier
        if avg_imm_msps >= MIN_IMMERSIVENESS_TO_PASS:
            reward *= np.clip(avg_imm_msps/MIN_IMMERSIVENESS_TO_PASS, 0, 1.2)
        else:
            # Severe penalty for not meeting minimum immersion
            reward *= 0.5
            
        if self.total_number_of_episodes_trained % 100 == 0:
            print(f"\033[92mAll imm: {all_imm}, Avg imm: {avg_imm_msps},reqs done: {reqs_fullfilled}, Minimum requests done: {minimum_requests_done}, Final reward: {reward}\033[0m")

        return reward

# I think it is going well, I need just to try and see what i get. 
# Maybe need to see what should be changed about the number of steps and what not.
