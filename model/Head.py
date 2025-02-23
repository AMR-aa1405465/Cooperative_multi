"""
This file contains the code for the Head class.
The Head class represent a cluster of users in the metaverse.
Each Head will have one room representing his virtual room.
"""
import numpy as np
from helpers import GlobalState
from helpers.Average import DynamicMovingAverage
from .VirtualRoom import VirtualRoom
import collections
from helpers.Constants import NO_MONEY_SPENT_FLAG


class Head:
    _head_id_counter = 0

    ################################### GENERAL FUNCTIONS ######################################
    def __init__(self, num_users: int, room: VirtualRoom, htype: int, target_immersiveness: float):
        self.id: int = Head._head_id_counter  # the id of the head
        Head._head_id_counter += 1
        self.room: VirtualRoom = room  # the virtual room of the head
        self.init_number_users = num_users
        self.num_users: int = num_users  # the number of users inside the head cluster.
        self.htype = htype  # HeadType.EXTERNAL or HeadType.LOCAL
        self.counter = 0  # used to index the hist dictionary
        self.hist = collections.OrderedDict()  # used to store the metrics of the head
        self.accumulate_help = {}
        self.check_head()
        self.target_immersion = target_immersiveness  # defines the target immersiveness for the head.
        self.number_of_done_requests = 0
        self.moving_average_immersion = DynamicMovingAverage()
        # todo: later on, this target immersirvness will change with time. 

    def check_head(self):
        """
        This function checks the initialization of the head.
        """
        assert self.id >= 0, f"Problem: id:{self.id} is not valid"
        assert self.num_users > 0, f"Problem: num_users:{self.num_users} is not valid"
        assert self.room is not None, f"Problem: room is not valid"
        assert self.htype is not None, f"Problem: type is not valid"
        assert len(self.hist) == 0, f"Problem: hist is not empty"

    def calculate_window_hist_metrics(self, window_size):
        """
        This function calculates the historical metrics of the head.
        """
        # get the last window size records from the hist dictionary
        last_window_size_records = list(self.hist.items())[-window_size:]
        # calculate the average cost and immersiveness
        total_cost = 0
        total_immersiveness = 0
        for key, value in last_window_size_records:
            total_cost += value['total_cost']
            total_immersiveness += value['immersiveness_score']
        return total_cost, total_immersiveness, len(last_window_size_records)

    def calculate_hist_metrics(self):
        """
        This function calculates the historical metrics of the head.

        """
        total_cost = 0
        total_immersiveness = 0
        for key, value in self.hist.items():
            total_cost += value['total_cost']
            total_immersiveness += value['immersiveness_score']
        return total_cost, total_immersiveness, len(self.hist)

    def increase_num_users(self, addional_users):
        """
        This function increases the number of users in the head.
        """
        self.num_users += addional_users

    ################################### Head-RELATED FUNCTIONS ######################################
    def get_accumulate_help_second_variant(self, head_id, percentage):
        """
        For testing purposes only.
        """
        if head_id == self.id:
            self.accumulate_help(percentage)

    def accumulate_help(self, percentage):
        """
        (executed only if the head does not have enought resources to serve its users)
        (executed first)
        This function accumulates of different msps's help percentage to cover the total cost @system_clock.
        After accumulating the help percentage, the head can serve its users with @percentage of the total cost.
        """
        system_clock = GlobalState.get_clock()
        if system_clock not in self.accumulate_help:
            self.accumulate_help[system_clock] = percentage
        else:
            self.accumulate_help[system_clock] += percentage
            assert self.accumulate_help[
                       system_clock] <= 1, f"Problem: accumulate_help[system_clock]:{self.accumulate_help[system_clock]} is not valid"

    # def apply_external_help(self):
    #     """
    #     (executed second)
    #     This function applies the external help that was accumulated earlier from accumulate_help to the head.
    #     """
    #     system_clock = GlobalState.get_clock()
    #     null_metrics = {
    #         'total_cost': 0,
    #         'immersiveness_score': 0,
    #     }
    #     if system_clock in self.accumulate_help:
    #         help_percentage = self.accumulate_help[system_clock]
    #         bitrate = self.room.max_bitrate * help_percentage
    #         frame_rate = self.room.max_frame_rate * help_percentage
    #         behavioral_accuracy = self.room.max_behavioral_accuracy * help_percentage
    #         return self.allocate_resources(action=[bitrate, frame_rate, behavioral_accuracy], add_to_hist_flag=True)
    #     else:
    #         print("No external help was applied")
    #         return self.allocate_resources(action=[NO_MONEY_SPENT_FLAG, NO_MONEY_SPENT_FLAG, NO_MONEY_SPENT_FLAG],
    #                                        add_to_hist_flag=True)

    def get_needed_costs(self):
        """
        this function returns 33%, 66% and 100% of the total cost needed for the virtual room with for this head.
        """
        percentages = [0.25, 0.5, 0.75, 1]
        costs = []
        for percentage in percentages:
            bitrate = self.room.max_bitrate * percentage
            frame_rate = self.room.max_frame_rate * percentage
            behavioral_accuracy = self.room.max_behavioral_accuracy * percentage
            metrics = self.room.calculate_metrics(given_bitrate=bitrate,
                                                  num_heads_clients=self.num_users,
                                                  behavioral_accuracy=behavioral_accuracy,
                                                  given_frame_rate=frame_rate)
            costs.append(metrics['total_cost'])
        return costs

        # return self.room.base_compute_units * self.num_users * 0.33, self.room.base_compute_units * self.num_users * 0.66, self.room.base_compute_units * self.num_users

    def allocate_resources(self, action, add_to_hist_flag):
        """
        This function is used to allocate the resources to the head.
        and return the satisfaction, immerviness and cost.
        """
        # we have four components in the action: Bitrate, frame rate, beahav accuracy. (Rotation speed is fixed)
        bitrate = np.interp(action[0], [0.25, 1], [self.room.min_bitrate, self.room.max_bitrate])
        frame_rate = np.interp(action[1], [0.25, 1], [self.room.min_frame_rate, self.room.max_frame_rate])
        behavioral_accuracy = np.interp(action[2], [0.25, 1],
                                        [self.room.min_behavioral_accuracy, self.room.max_behavioral_accuracy])
        metrics = None
            # apply the action to the room
        metrics = self.room.calculate_metrics(given_bitrate=bitrate,
                                                num_heads_clients=self.num_users,
                                                behavioral_accuracy=behavioral_accuracy,
                                                given_frame_rate=frame_rate)

        if add_to_hist_flag:
            assert self.counter not in self.hist, f"Problem: Key:{self.counter} already in hist"
            self.hist[self.counter] = metrics
            self.counter += 1

        return metrics['immersiveness_score'], metrics['total_cost'], metrics

    def add_to_hist(self, metrics):
        """
        This function adds the metrics to the history.
        """
        assert self.counter not in self.hist, f"Problem: Key:{self.counter} already in hist"
        self.hist[self.counter] = metrics
        self.counter += 1

    def get_target_immersiveness(self):
        # todo: later on, this target immersirvness will change with time. 
        # if GlobalState.get_clock() % 100 == 0: 
        return self.target_immersion
        ###################################### Simulation-RELATED FUNCTIONS ##################################

    def episode_reset(self):
        """
        This function resets the head for the episodes.
        """
        self.counter = 0  # timestep counter to 0
        self.hist.clear()  # removes the history.
        # self.num_users = self.init_number_users  # returns original number of users. todo: open again if needed. 
        self.accumulate_help = {}
        self.number_of_done_requests = 0
        self.moving_average_immersion.reset()

        # self.check_head()

    def decrease_num_users(self, substracted_users):
        """
        This function decreases the number of users in the head.
        """
        self.num_users -= substracted_users

    # def estimate_peak_resource_needs(self):
    #     """
    #     Estimate peak resource requirements based on room parameters and user count
    #     """
    #     return self.room.calculate_metrics(
    #         given_bitrate=self.room.max_bitrate,
    #         num_heads_clients=self.num_users,
    #         behavioral_accuracy=self.room.max_behavioral_accuracy,
    #         given_frame_rate=self.room.max_frame_rate
    #     )
    #
    # def get_performance_gaps(self):
    #
    #     """
    #     Calculate gaps between current and ideal performance
    #     """
    #     if not self.hist:
    #         return None
    #
    #     current = list(self.hist.values())[-1]
    #     ideal = self.estimate_peak_resource_needs()
    #
    #     return {
    #         'immersiveness_gap': ideal['immersiveness_score'] - current['immersiveness_score'],
    #         'cost_gap': ideal['total_cost'] - current['total_cost']
    #     }
