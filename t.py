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
        self.name = "random"
        # env = GameCoopEnv(run_name=self.name,max_clock=200,msps_requests=10)
        # env = GameCoopEnv(run_name=self.name,max_clock=200,msps_requests=30)
        # env = GameCoopEnv(run_name=self.name,max_clock=200,msps_requests=50)
        # env = GameCoopEnv(run_name=self.name,max_clock=200,msps_requests=70)
        env = GameCoopEnv(run_name=self.name, max_clock=200, msps_requests=50)
        # to complete the development 
        # env.reset()
        # env.reset()
        # env.reset()
        # check_env(env)

        # print("-----------------------------------------------")
        for episode in range(1):
            env.reset()
            done = False
            rewards = []
            while not done: 
                sample_action = env.action_space.sample()
                # user_sample = input(f"enter an action of length {env.num_msps}: ")
                # sample_action = [int(i) for i in user_sample.split(" ")]
                # sample_action = [3 for _ in range(env.num_msps)]
                # sample_action = [63,63,63]
                # sample_action = [1,1,1]
                # sample_action = [63,63,63]
                # sample_action = [31,47,63]
                # sample_action = [31,31,31]
                # sample_action = [15,15,15]
                # sample_action = [10  for _ in range(env.num_msps)]
                # sample_action = [12  for _ in range(env.num_msps)]

                # sample_action = [21,21,21]
                # sample_action = [0,0]
                # print(f"@{self.name}, Info: Sample action: {sample_action}")
                state, reward, done, trunc, info = env.step(sample_action)
                rewards.append(reward)
                decoded_actions = env.parse_action(sample_action, operation_mode=0)
                # print(f"@{self.name}, Info: State: {state}, Reward: {reward}, Done: {done}")
                # print(f"@{self.name}, Info: Reward: {reward}, Done: {done}\n")
                if done:
                    print("Final state",state)
                    # print(msp.achvd_immers for msp in env.msp_list)
                    print("total acculmated reward: ", sum(rewards))
                # input("onetimestep completed")

            print("=" * 100, end="\n\n")
            # input("Step?")
            # env.render()

        #     if done:
        # env.render()
        #         # print(f"@{self.name}, Info: Episode finished with reason: {info['reason']}")
        #         input("dd")


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
