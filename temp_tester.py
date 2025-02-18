# from helpers.GlobalState import GlobalState

# print(GlobalState.get_clock())

# GlobalState.increment_clock()
# print(GlobalState.get_clock())


# def normalize_value(value: float, min_value: float, max_value: float) -> float:
#     """
#     Normalize a value to be between -1 and 1.

#     :param value: The value to be normalized.
#     :param min_value: The minimum value of the range.
#     :param max_value: The maximum value of the range.
#     :return: The normalized value between -1 and 1.
#     """
#     if min_value == max_value:
#         raise ValueError("min_value and max_value cannot be the same")

#     normalized = 2 * ((value - min_value) / (max_value - min_value)) - 1
#     assert normalized >= -1 and normalized <= 1, "Normalized value is out of bounds"
#     return normalized

# test0 = normalize_value(0, 0, 1)
# test1 = normalize_value(0, 0, 100)
# test2 = normalize_value(25, 0, 100)
# test3 = normalize_value(50, 0, 100)
# test4 = normalize_value(75, 0, 100)
# test5 = normalize_value(100, 0, 100)
import time
from EnvCooperation import GameCoopEnv
from gymnasium.utils.env_checker import check_env

# import streamlit as st

# Initialize Streamlit placeholders for dynamic updates
# st.set_page_config(layout="wide")
# status_placeholder = st.empty()

# DRL Loop
# print(test0, test1, test2, test3, test4, test5)
class Test:
    def __init__(self):
        self.name = "test3"
        env = GameCoopEnv(run_name=self.name,max_clock=200,msps_requests=90)
        maximum_settings = {"action":-1,"reward":-1000,"decoded_actions":[]}
        # to complete the development 
        # env.reset()
        # env.reset()
        # env.reset()
        # check_env(env)
        # actions_list = [i for i in range(64)]
        # actions_list = [10 for i in range(64)]
        actions_list = [10]
        for action in actions_list:
        # print("-----------------------------------------------")
            for episode in range(1):
                env.reset()
                done = False
                reward =0 
                while not done:
                    # sample_action = env.action_space.sample()
                    sample_action = [action for _ in range(env.num_msps)]
                    state, reward, done, trunc, info = env.step(sample_action)
                    decoded_actions = env.parse_action(sample_action, operation_mode=0)
                print("action: ", action,"Final reward was ", reward)   
                if reward > maximum_settings["reward"]:
                    maximum_settings["reward"] = reward
                    maximum_settings["action"] = action
                    maximum_settings["decoded_actions"] = decoded_actions
                print("="*100,end="\n\n")
        print("Maximum reward was ", maximum_settings["reward"], "with action ", maximum_settings["action"])
        print("Decoded actions: ", maximum_settings["decoded_actions"])
test = Test()

# print(env.observation_space.shape)


"""
+-----------------+
| GameCoopEnv     |
+-----------------+
| - run_name      |
| - max_clock     |
| - vrooms_list   |
| - msp_list      |
| - head_list     |
| - state         |
| - observation_space |
| - action_space  |
+-----------------+
| + __init__()    |
| + step()        |
| + reset()       |
| + render()      |
| + create_virtual_room() |
| + create_heads()|
| + create_msps() |
| + normalize_value() |
| + is_done()     |
| + get_state()   |
| + parse_set_action() |
| + apply_decoded_action() |
| + calculate_reward() |
+-----------------+
        |
        | uses
        v
+-----------------+
| VirtualRoom     |
+-----------------+
| - id            |
| - max_users     |
| - user_scale_compute |
| - user_scale_bandwidth |
| - user_density_factor |
| - resource_sharing_factor |
| - polygon_count |
| - physics_objects |
| - interaction_points |
| - num_sensors   |
| - state_variables |
| - update_frequency |
| - min_bitrate   |
| - max_bitrate   |
| - min_frame_rate |
| - max_frame_rate |
| - min_structural_accuracy |
| - max_structural_accuracy |
| - rotation_speed |
| - quality_weights |
| - base_compute_units |
| - user_scaling_factor |
| - unit_bandwidth_cost |
| - unit_compute_cost |
| - min_behavioral_accuracy |
| - max_behavioral_accuracy |
| - ssim_coeff    |
| - vmaf_coeff    |
+-----------------+
| + __init__()    |
| + post_init_checker() |
| + calculate_metrics() |
+-----------------+

+-----------------+
| MSP             |
+-----------------+
| - id            |
| - heads         |
| - budget        |
| - actions_generated |
+-----------------+
| + get_possible_actions() |
+-----------------+
        |
        | contains
        v
+-----------------+
| Head            |
+-----------------+
| - num_users     |
| - room          |
| - type          |
+-----------------+
| + ...           |
+-----------------+

+-----------------+
| GlobalState     |
+-----------------+
| + increment_clock() |
| + reset_clock() |
| + get_clock()   |
+-----------------+
"""