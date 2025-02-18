import os
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from EnvCooperation import GameCoopEnv
import glob

RUNNAME = "loaded"
# MSPS_REQUESTS = 30
n_procs = 10


def enviroment_generator(RUNNAME, MSPS_REQUESTS, extra_info):
    # myseed = random.randint(0, 100)
    # print("Seed =", myseed)
    # envi = NSEnv(seed=myseed, runname=RUNNAME, file_name=RUNNAME)
    envio = GameCoopEnv(run_name=RUNNAME, max_clock=600, msps_requests=MSPS_REQUESTS, extra_info=extra_info)
    return envio


# My best results (moved to results/27jan)
# files = sorted(glob.glob("./results/msps_5_requests_*_gamma_0.97_check_rew4_oldintermediate_msps_5"))
# files = sorted(glob.glob("./results/msps_5_requests_*_gamma_0.97_check_rew4_newintermediate_msps_5"))

# files = sorted(glob.glob("./results/msps_5_requests_*_gamma_0.97_check_rew4_oldintermediate_msps_5_changedroomcost"))
# files = sorted(glob.glob("./results/msps_1_requests_*_gamma_0.97_non_cooper_msps_1"))
# files = sorted(glob.glob("./results/msps_5_requests_*_gamma_0.97_non_cooper_msps_5"))
# files = sorted(glob.glob("./results/msps_5_requests_*_gamma_0.97_after_changing_reward"))

# files = sorted(glob.glob("./results/msps_5_requests_90_gamma_0.97_after_changing_reward2"))
# files = sorted(glob.glob("./results/msps_5_requests_90_gamma_0.97_after_changing_and_immediate_avg"))
# files = sorted(glob.glob("./results/msps_5_requests_90_gamma_0.97_after_changing_and_immediate_avg"))
# files = sorted(glob.glob("./results/msps_5_requests_90_gamma_0.97_after_changing_and_immediate"))
# files = sorted(glob.glob("./results/msps_3_requests_*_gamma_0.97_limiting_by_0100"))
# files = sorted(glob.glob("./results/msps_3_requests_*_gamma_0.97_limiting_by_0100"))
files = sorted(glob.glob("./results/msps_3_requests_*_gamma_0.97_limiting_by_0180"))
# files = sorted(glob.glob("./results/msps_5_requests_*_gamma_0.97_after_changing_rewardPPO"))

# //results/msps_5_requests_30_gamma_0.97_check_rew4_newintermediate_msps_5

for f in files:
    print(f)
    # print(f.split("_"))

    # requests = int(f.split("_")[5]) # for results in the project_runs_18thjan folder.
    requests = int(f.split("_")[3])  # for local results.
    env = GameCoopEnv(run_name=RUNNAME, max_clock=1000, msps_requests=requests, extra_info=f[-10:])
    # load_path = "/Users/mac/Documents/Manual Library/project_runs_18thjan/results/msps_5_requests_30_gamma_0.97_newrunner/best_model/best_model"
    load_path = f"{f}/best_model/best_model"
    task = SubprocVecEnv(
        [lambda: Monitor(enviroment_generator(RUNNAME=RUNNAME, MSPS_REQUESTS=requests, extra_info=f[-10:])) for _ in
         range(n_procs)], start_method='fork')
    # model = PPO.load(load_path)
    model = A2C.load(load_path)
    model.set_env(task)  # used to set the enviroment for multiprocessing
    # model.ent_coef = 0.1
    done = False
    obs, info = env.reset()
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        print(action)
        obs, rewards, done, truncated, info = env.step(action)
    print("*" * 100)
