"""
This file contains the code for the MSP class.
The MSP class represent a metaverse service provider. 
An MSP can serve multiple Heads, and each Head can have multiple users.
Each Head will have one room representing his virtual room.
The action for the msp is given *currently* by the DRL agent. 
I need to create a function that will create a list of possible actions for the msp 
according to the number of heads it has. 
"""
import itertools
from typing import List, Tuple

import numpy as np

from helpers import Constants
from .Head import Head
from helpers.GlobalState import GlobalState
from helpers.Constants import (ACTION_DOABLE_AND_APPLIED, ACTION_NOT_DOABLE, LOWER_BUDGET_THRESHOLD, PENALITY_WEIGHT,
                               UNIVERSAL_POSSIBLE_PERCENTAGES, IMMERSIVNESS_FREEDOM)


def calculate_penality(x_target, x_current, mode="linear"):
    x_target = np.clip(x_target, 0, 1)
    x_current = np.clip(x_current, 0, 1)
    difference = abs(x_target - x_current)
    head_penality_wi_clip = 0
    if mode == "exp":
        difference = abs(x_target - x_current)
        # head_penality_wo_clip = Constants.PENALITY_WEIGHT * (np.exp(difference) - 1)
        head_penality_wi_clip = np.clip(Constants.PENALITY_WEIGHT * (np.exp(difference) - 1), 0, 1)
        head_penality_wi_clip = round(head_penality_wi_clip, 2)
    elif mode == "linear":
        difference = abs(x_target - x_current)
        # head_penality_wo_clip = Constants.PENALITY_WEIGHT * difference
        head_penality_wi_clip = np.clip(Constants.PENALITY_WEIGHT * difference, 0, 1)
        head_penality_wi_clip = round(head_penality_wi_clip, 2)

    elif mode == "linear_sqr":
        difference = abs(x_target - x_current)
        # head_penality_wo_clip = Constants.PENALITY_WEIGHT * difference
        head_penality_wi_clip = np.clip(Constants.PENALITY_WEIGHT * difference ** 2, 0, 1)
        head_penality_wi_clip = round(head_penality_wi_clip, 2)

    elif mode == "sigmoid":
        k = 10  # steepness
        midpoint = 0.2
        penalty = 1 / (1 + np.exp(-k * (difference - midpoint)))
        head_penality_wi_clip = abs(penalty)
    return head_penality_wi_clip


class MSP:
    _msp_id_counter = 0

    ################################### GENERAL FUNCTIONS ######################################
    def __init__(self,
                 budget: float,
                 heads: List[Head],
                 num_requests: int,
                 # heads_target_imm: float,
                 neighbors = []
                 ):
        self.id: int = MSP._msp_id_counter
        # self.heads_target_imm = heads_target_imm
        MSP._msp_id_counter += 1
        self.computed_before = False
        self.heads: List[Head] = heads
        self.accumulate_help = {}  # THE help received by other MSPs at a certain point in time.
        self.budget = budget
        self.neighbors: List[MSP] = []
        # New :
        self.num_requests = num_requests  # this defines how many requests should be fillfilled
        self.num_requests_fullfilled = 0  # this defines how many requests have been filled (takes a decimal number)
        self.num_requests_done = 0  # this just counts how many of the requests have been done. (i.e., the msp had enought budget to do.)
        self.final_reward = 0
        self.b = 0
        self.b_avg = 0
        self.achvd_immers = 0
        #
        self.initial_budget = budget
        self.total_help_received = 0  # the total help received by this msp for a whole episode.
        self.total_helped_times = 0  # the total times this msp helped other msps.
        self.actions_generated = False
        self.actions_dict, self.number_of_actions = self.get_possible_actions()
        # added new : 
        self.help_percentages, self.num_help_perc = None, None # to be set manually. 
        self.heads_history_struct = self.generate_heads_history()
        self.check_msp()
        # used for the env state, if number of clinets increases.
        self.initial_num_clients = self.get_total_num_clients()

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)
        self.help_percentages, self.num_help_perc = self.get_possible_help_actions()

    def generate_heads_history(self):
        """
        This function will generate the history of the heads.
        """
        s = {
            'timestep': [],
            'help_received_with_time': []
        }
        for head in self.heads:
            s[f'{head.id}_immersiveness'] = []
            s[f'{head.id}_cost'] = []
        # s['help_received_with_time'] = []
        return s

    def check_msp(self):
        """
        This function will check the msp & its heads.
        """
        assert len(self.heads) > 0, "MSP must have at least one head"
        assert self.budget > 0, "MSP must have a budget"
        assert len(self.heads_history_struct['timestep']) == 0, "MSP must be empty at first."
        assert len(self.accumulate_help) == 0, "MSP must be empty at first."

    def get_heads_history(self):
        return self.heads_history_struct

    def need_help(self):
        """
        This function will return True if the msp has requested help from other msps.
        """
        return self.budget <= 0 or self.budget <= (self.initial_budget * LOWER_BUDGET_THRESHOLD)
    
    def get_budget_percentage(self):
        """
        This function will return the percentage of the budget that the msp has used.
        """
        return self.budget / self.initial_budget

    def is_msp_finished_budget(self):
        """
        This function will check if the msp has finished its budget.
        @return true or false, and a message indicating the reason.
        @note: the least possible action is 0.25,0.25,0.25,0.25 for its heads.
        """
        # TODO : Maybe need to optimizate this function as I am calling it in multiple places.

        # first check if budget <= 0 .
        if self.budget <= 0:
            return True, f"msp {self.id} action not possible, Budget is 0"
        # Second check if msp can perform the least possible action 0.25,0.25,0.25,0.25 for its heads
        least_possible_perc = min(UNIVERSAL_POSSIBLE_PERCENTAGES)
        heads_actions = [(least_possible_perc, least_possible_perc, least_possible_perc) for _ in self.heads]
        res_ = self.perform_mock_heads_action(heads_actions, debug=False)
        if self.budget < res_["total_cost"]:
            self.budget = 0  # just to speed things up.
            return True, f"msp {self.id} action not possible, budget < least possible action"
        return False, f"msp {self.id} action possible, Budget is {self.budget}"

    def get_total_num_clients(self):
        return sum(head.num_users for head in self.heads)

    def add_head(self, head: Head):
        """
        This function will be used (In case the MSP wanted to add global-heads)
        The global Heads are the ones that are global and not attached to any of the msps yet.
        """
        self.actions_generated = False  # reset the actions generated flag.
        self.heads.append(head)  # add the head
        self.heads_history_struct = self.generate_heads_history()  # Generate the heads history struct.
        self.get_possible_actions()  # generate the possible actions for the msp.

    def get_budget(self):
        return self.budget

    def get_num_heads(self):
        return len(self.heads)
    
    def increase_budget(self, amount):
        """
        This function will increase the budget of the msp by the amount given unless it exceeds the initial budget.
        """
        self.budget = min(self.budget + amount, self.initial_budget)

    def decrease_budget(self, amount):
        """
        This function will decrease the budget of the msp by the amount given.
        """
        self.budget = max(self.budget - amount, 0)
        
    ################################### MSP-RELATED FUNCTIONS ######################################

    def aggregate_res_budget(self, money_received):
        # Keep track of the help received by this msp at a certain point in time.

        system_clock = GlobalState.get_clock()

        if self.accumulate_help.get(system_clock, 0) == 0:
            self.accumulate_help[system_clock] = money_received
            # self.heads_history_struct['help_received_with_time'].append(money_received)
        else:
            self.accumulate_help[system_clock] += money_received
            # val = self.heads_history_struct['help_received_with_time'].pop()
            # self.heads_history_struct['help_received_with_time'].append(val + budget_percentage)
    
        # there is no need for that.
        # # update the heads with the help received.
        # for head in self.heads:
        #     head.accumulate_help(budget_percentage)

    # def get_total_payment_per_percentage(self):
    #     """
    #     This function will return the total payment for the 33%, 66% and 100% of the total cost needed for 
    #     the virtual room with for this msp.
    #     (to be used when this msp gets helped by others so that we can split the payment of this msp among helpers.)
        
    #     @return: (quarter_payments 25% of total cost, half_payments 50% of total cost, three_quarter_payments 75% of total cost, full_payments 100% of total cost)
    #     of all the heads this msp serves.
    #     """
    #     quarter_payments = []
    #     half_payments = []
    #     three_quarter_payments = []
    #     full_payments = []
    #     for head in self.heads:
    #         quarter, half, three_quarter, full = head.get_needed_costs()
    #         quarter_payments.append(quarter)
    #         half_payments.append(half)
    #         three_quarter_payments.append(three_quarter)
    #         full_payments.append(full)
    #     return sum(quarter_payments), sum(half_payments), sum(three_quarter_payments), sum(full_payments)

    # def apply_external_budget(self):
    #     """
    #     This function will apply the external budget to the heads after the budget is aggregated.
    #     the budget is aggregated by @aggregate_res_budget function. 
    #     """
    #     system_clock = GlobalState.get_clock()
    #     self.heads_history_struct['timestep'].append(system_clock)
    #     for head in self.heads:
    #         immerivenss, cost = head.apply_external_help()
    #         self.heads_history_struct[f'{head.id}_immersiveness'].append(immerivenss)
    #         self.heads_history_struct[f'{head.id}_cost'].append(cost)
    
    def get_possible_help_actions(self):
        """
        Generate all possible combinations of help values for neighbors
        Each neighbor can have help values from 0.1 to 0.9 in steps of 0.1
        Returns a list of lists where each inner list contains dictionaries with index and help values
        """
        if self.computed_before:
            return self.combinations_dict, self.combinations_num

        combinations = []
        help_values = np.arange(0, 0.8, 0.2)
        
        # Generate all possible combinations of help values for each neighbor
        num_neighbors = len(self.neighbors)
        for help_values_combination in itertools.product(help_values, repeat=num_neighbors):
            combination = []
            for help_value in help_values_combination:
                combination.append(help_value)
            combinations.append(combination)
        
        self.computed_before = True
        self.combinations_dict = dict(enumerate(combinations))
        self.combinations_num = len(combinations)
        return self.combinations_dict, self.combinations_num


    ################################## SIMULATION-RELATED FUNCTIONS ##################################  
    def get_possible_actions(self):
        """
        This function returns all possible combinations of resource allocations for all heads.
        Each head can have different combinations of [bitrate, framerate, behavioral_accuracy].
        The bitrate, framerate and behavioral_accuracy are given as percentages of the maximum values.
        Returns:
            List[List[Tuple]]: List of all possible combinations for all heads
            int: Total number of possible actions
        """
        if self.actions_generated:
            return self.actions_dict, self.number_of_actions

        def generate_head_combinations(head_index, current_combination):
            # Base case: if we've assigned resources to all heads
            if head_index == len(self.heads):
                return [current_combination[:]]

            combinations = []
            # For each head, try all possible combinations of resources
            for max_bit_rate_perc in UNIVERSAL_POSSIBLE_PERCENTAGES:
                for max_frame_rate_perc in UNIVERSAL_POSSIBLE_PERCENTAGES:
                    for max_behav_acc_perc in UNIVERSAL_POSSIBLE_PERCENTAGES:
                        # Append the current combination for this head
                        current_combination.append((max_bit_rate_perc, max_frame_rate_perc, max_behav_acc_perc))
                        # Recurse to generate combinations for the next head
                        combinations.extend(generate_head_combinations(head_index + 1, current_combination))
                        # Remove the last added combination to backtrack
                        current_combination.pop()

            return combinations

        # Generate all possible combinations starting with an empty list
        actions_dict = generate_head_combinations(0, [])
        number_of_actions = len(actions_dict)
        self.actions_generated = True
        my_dict = {index: value for index, value in enumerate(actions_dict)}
        return my_dict, number_of_actions

    def episode_reset(self):
        self.budget = self.initial_budget
        self.accumulate_help.clear()
        self.total_help_received = 0
        self.total_helped_times = 0
        self.num_requests_fullfilled = 0
        self.num_requests_done = 0
        self.heads_history_struct.clear()
        self.b = 0
        self.b_avg = 0
        self.achvd_immers = 0
        self.heads_history_struct = self.generate_heads_history()

    def check_apply_msp_action(self, action_tuple_list):
        """
        This function will check if the action is doable by the msp or not.
        and returns the total_cost, total_immersiveness anyway.

        @param: action_tuple_list: a list of tuples, each tuple is a list of 3 numbers representing the action for a head.
        @return: (0,1), total_cost, avg_total_immersiveness.

        @note: 0 means the action is not applied, 1 means the action is applied.
        """

        heads_actions = [all_heads_actions for all_heads_actions in action_tuple_list]
        print("heads_actions=", heads_actions)
        # TODO: implement the external help here (sure in-sha-allah).
        mock_res = self.perform_mock_heads_action(heads_actions, debug=True)
        self.get_cost_per_action()

        res ={}
        if mock_res["total_cost"] <= self.budget:
            self.commit_action(mock_res["temp_list"], mock_res["total_cost"], mock_res["total_immersiveness"], mock_res["num_satisfied_requests"])
            self.add_to_heads(mock_res["penality_satisifaction_list"], True)
            
            res["action_flag"] = ACTION_DOABLE_AND_APPLIED
            res['total_cost'] = mock_res["total_cost"]
            res['total_immersiveness'] = mock_res["total_immersiveness"]
            res['num_satisfied_requests'] = mock_res["num_satisfied_requests"]
            res['penalties'] = mock_res["penalities"]
            res['penalty_satisfaction_list'] = mock_res["penality_satisifaction_list"]
            return res
        else:
            # TODO: implement the external help here. (not sure)
            self.add_to_heads(mock_res["penality_satisifaction_list"], False)
            res["action_flag"] = ACTION_NOT_DOABLE
            res['total_cost'] = mock_res["total_cost"]
            res['total_immersiveness'] = mock_res["total_immersiveness"]
            res['num_satisfied_requests'] = mock_res["num_satisfied_requests"]
            res['penalties'] = mock_res["penalities"]
            res['penalty_satisfaction_list'] = mock_res["penality_satisifaction_list"]
            return res

    def add_to_heads(self, penalty_satisfaction_list, positive_hist_flag):
        """
        This function will add the penalty_satisfaction_list to the heads.
        positive flag indicates if the msp done the action or not. 
        """

        for rec in penalty_satisfaction_list:
            head_index = rec["index"]
            if positive_hist_flag:
                self.heads[head_index].number_of_done_requests += 1
                clipped_immersiveness = np.clip(rec["details"]["current_immersiveness"], 0, 1)
                self.heads[head_index].moving_average_immersion.append(clipped_immersiveness)
            else:
                self.heads[head_index].moving_average_immersion.append(0)
    
    def perform_mock_heads_action(self, heads_actions, debug=False):
        """
        simulates the application of the action taken by the msp to all of its heads.
        @Return: total_cost required for all of their actions, the total immersiveness gained from this action.
        """
        total_cost = 0
        total_immersiveness = 0
        temp_list = []  # used by another function.
        penality_satisifaction_list = []  # used by another function.
        # immersiveness_satisfaction =0 
        penalities = 0
        num_satisfied_requests = 0
        for i in range(len(self.heads)):
            # applies the action for each of the heads in that MSP.
            immerivenss, cost, metrics = self.heads[i].allocate_resources(heads_actions[i], add_to_hist_flag=False)
            assert 0 <= immerivenss <= 1, "Immersiveness should be between 0 and 1"
            total_cost += cost
            total_immersiveness += immerivenss

            target_immersiveness = self.heads[i].get_target_immersiveness()
            penalty_mode = "sigmoid"
            if immerivenss >= IMMERSIVNESS_FREEDOM * target_immersiveness:
                num_satisfied_requests += 1
                penalty = calculate_penality(target_immersiveness, immerivenss, mode=penalty_mode)
            else:
                penalty = calculate_penality(target_immersiveness, immerivenss, mode=penalty_mode)

            penalities += penalty
            penality_satisifaction_list.append({
                "id": self.heads[i].id,
                "index": i,
                "cost": cost,
                "penality": penalty,
                "details": {
                    "target_immersiveness": target_immersiveness,
                    "current_immersiveness": immerivenss
                }
            })

            temp_list.append({"id": self.heads[i].id, "metrics": metrics})  # this is for the history.  
        return {
            "total_cost": total_cost,
            "total_immersiveness": total_immersiveness,
            "temp_list": temp_list,
            "num_satisfied_requests": num_satisfied_requests,
            "penalities": penalities,
            "penality_satisifaction_list": penality_satisifaction_list
        }

    def commit_action(self, temp_list, total_cost, total_immersiveness, num_satisfied_requests):
        """
        Apply the valid action to the heads and update the history.
        """
        system_clock = GlobalState.get_clock()
        self.heads_history_struct['timestep'].append(system_clock)
        for i, head in zip(range(len(self.heads)), temp_list):
            self.heads[i].add_to_hist(head["metrics"])
            self.heads_history_struct[f'{self.heads[i].id}_immersiveness'].append(
                head["metrics"]['immersiveness_score'])
            self.heads_history_struct[f'{self.heads[i].id}_cost'].append(head["metrics"]['total_cost'])

        self.budget -= total_cost  # updates the msp budget here.
        self.heads_history_struct['help_received_with_time'].append(0)

        if num_satisfied_requests == len(self.heads):  # all of them are fullfilled.
            self.num_requests_fullfilled += 1



    def get_cost_per_action(self):
        """
        In this function, either the action sequence is given or the action number is given.
        an example of the action_seq is : [(1,1,1),(0.25,0.25,0.25)] for 2 heads. 
        but almost in current version, I will use it with only one heads [(1,1,1)]
        """
        heads_actions_hundered = [[1,1,1] for _ in range(len(self.heads))]
        heads_actions_25 = [[0.25,0.25,0.25] for _ in range(len(self.heads))]
        heads_actions_50 = [[0.5,0.5,0.5] for _ in range(len(self.heads))]
        heads_actions_75 = [[0.75,0.75,0.75] for _ in range(len(self.heads))]
        # later on, I can use a list of actions and choose the highest one of them according to the summation of the accumulated help.

        # checking the cost of 1, 0.75, 0.5, 0.25
        mock_hundered = self.perform_mock_heads_action(heads_actions_hundered, debug=True)
        mock_25 = self.perform_mock_heads_action(heads_actions_25, debug=True)
        mock_50 = self.perform_mock_heads_action(heads_actions_50, debug=True)
        mock_75 = self.perform_mock_heads_action(heads_actions_75, debug=True)
        # can add more combinations here.

        # returns a dict of the cost for each of the configuration.
        result = {
            0.25 : mock_25["total_cost"], # the minimum cost
            0.5 : mock_50["total_cost"], 
            0.75 : mock_75["total_cost"],
            1 : mock_hundered["total_cost"] # the maximum cost
        }
        print(f"@Info:{self.__class__.__name__}, result={result}")
        return result


        
# msp flow :
# 1. the msp is intialized with the heads and budeg
# 2. it generates the actions for the heads
# 3. the msp will either (take an action from the DRL agent or get external help ) 
# (both have different funcs)
# 3.1 if received external help, it will first accumulate the help received, then apply it.
# 3.2 if the msp takes an action from drl, it will apply it to the heads.
