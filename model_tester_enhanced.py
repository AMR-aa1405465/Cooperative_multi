import os
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from EnvCooperation import GameCoopEnv
import glob

# RUNNAME = "loaded"
n_procs = 10


def enviroment_generator(RUNNAME, MSPS_REQUESTS, extra_info, msps_count):
    extra_info = f"msps_count_{msps_count}_num_requests_{MSPS_REQUESTS}"
    envio = GameCoopEnv(run_name=RUNNAME, max_clock=1000, msps_requests=MSPS_REQUESTS, extra_info=extra_info)
    return envio


files = sorted(glob.glob("./results/msps_5_requests_*_gamma_0.97_after_changing_and_immediate_avg"))

# msps_1_requests_10_gamma_0.97_non_cooper_msps_1

for f in files:
    requests = int(f.split("_")[3])  # for local results.
    num_msps = 5
    print(f"@{num_msps}, Info: Requests: {requests}")
    RUNNAME = f"DRL_{num_msps}_MSPS"
    env = GameCoopEnv(run_name=RUNNAME, max_clock=1000, msps_requests=requests, extra_info=f[-10:],)
    load_path = f"{f}/best_model/best_model"
    task = SubprocVecEnv([lambda: Monitor(
        enviroment_generator(RUNNAME=RUNNAME, MSPS_REQUESTS=requests, extra_info=f[-10:], msps_count=num_msps)) for _ in
                          range(n_procs)], start_method='fork')
    model = A2C.load(load_path)
    model.set_env(task)  # used to set the enviroment for multiprocessing
    done = False
    obs, info = env.reset()
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = env.step(action)
    print("*" * 100)
