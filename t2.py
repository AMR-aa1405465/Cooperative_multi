import time
from EnvCooperation import GameCoopEnv
from gymnasium.utils.env_checker import check_env


class Test:
    def __init__(self):

        for num_requests in range(10, 100, 20):
            msps_count = 1 # make sure that you change the number of added msps in EnvCooperation.py file.
            # self.name = f"optimal_fixed_f{msps_count}"
            self.name = f"avg_fixed_f{msps_count}"
            # self.name = f"random_fixed_f{msps_count}"
            extra_info = f"msps_count_{msps_count}_num_requests_{num_requests}"
            env = GameCoopEnv(run_name=self.name, max_clock=200, msps_requests=num_requests, extra_info=extra_info)
            # print("-----------------------------------------------")
            for episode in range(1):
                env.reset()
                done = False
                while not done:
                    # sample_action = [10 for _ in range(env.num_msps)]
                    # sample_action = [0 for _ in range(env.num_msps)] # for saving.
                    # sample_action = [63 for _ in range(env.num_msps)] # for saving.
                    sample_action = [21 for _ in range(env.num_msps)] # for saving.
                    state, reward, done, trunc, info = env.step(sample_action)  # input("dd")
                    print(f"@{self.name}, Info: State: {state}, Reward: {reward}, Done: {done}")
                print("=" * 100, end="\n\n")


test = Test()
