import argparse
import sys
import signal
import sys
import gymnasium as gym
from gymnasium import Env

import numpy as np
import random
import os
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3 import PPO, A2C, TD3, SAC
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from torch.distributed.argparse_util import env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import os
from stable_baselines3.common.env_util import make_vec_env
from RL_trainer import RLTrainer

# from envMDAction import GameCoopEnv
from EnvCooperation import GameCoopEnv

envi = None
model = None

# Create the parser
my_parser = argparse.ArgumentParser(description='List the content of a folder')

######################### Add the arguments #########################

# Train or test
my_parser.add_argument('--train', type=bool, default=True)

my_parser.add_argument('--runname', type=str, required=True)

my_parser.add_argument('--ls', type=int, default=2500000)

my_parser.add_argument('--msps_requests', type=int, required=True)

my_parser.add_argument('--gamma', required=True, type=float, default=0.9)

my_parser.add_argument('--alg', default='ppo', type=str)

my_parser.add_argument('--bsize', default=int(2e6), type=int)

my_parser.add_argument('--load_model', type=bool, default=False)

my_parser.add_argument('--loaded_model_run', type=str, required=False)

my_parser.add_argument('--msg', type=str, required=True)

args = my_parser.parse_args()

RUNNAME = args.runname
GAMMA = args.gamma
MAX_STEPS = args.ls
ALG = args.alg
BUFFER_SIZE = args.bsize
LOAD = args.load_model
LOADED_MODEL_RUN = args.loaded_model_run
MSG = args.msg
# msps requests, added specifically for the experiment. 
MSPS_REQUESTS = args.msps_requests
data = {
    "BUFFER_SIZE": BUFFER_SIZE,
    "RUNNAME": RUNNAME,
    "GAMMA": GAMMA,
    "MAX_STEPS": MAX_STEPS,
    "Alg": ALG,
    "load": LOAD,
    "msg": MSG,
    "MSPS_REQUESTS": MSPS_REQUESTS
}

# Export the configuration to a file
# insert the date and time to the configurations.txt
import datetime

# os.system(f"cp Constants.py envMDActionSpace.py envMDAction.py ./results/{RUNNAME}")
# create a function to append to file , the function takes filename and content and append them 
def append_to_file(filename, content):
    if not os.path.exists(filename):
        # create the file if it does not exist
        
        with open(filename, 'w') as file:
            file.write(content)
    else:
        with open(filename, 'a') as file:
            file.write(content)

append_to_file("experiments_sequence.txt", "Hello, world!")



def handler(signum, frame):
    try:
        print('Signal handler called with signal', signum)
        print("Saving the model right Now Boss")

        model_path = f"./results/{RUNNAME}/current_model"
        print(f"Saving the model in {model_path}")
        #model.set_env()
        #model.save(model_path)
        model.save_model(model_path)
    except Exception as e:
        print("Error happend ", e, e.args)

    # stats_path = os.path.join(f"./results/{RUNNAME}", "vec_normalize.pkl")
    #
    # if LOAD:
    #     envx.save(stats_path)
    # else:
    #     env.save(stats_path)

    choice = input("Do you want to terminate??")
    if (choice.lower() == "y"):
        envi.close()
        sys.exit(0)


# regisstering the signal handler
# signal.signal(signal.SIGINT, handler)


# return an enviroment with random seed
def enviroment_generator():
    # myseed = random.randint(0, 100)
    # print("Seed =", myseed)
    # envi = NSEnv(seed=myseed, runname=RUNNAME, file_name=RUNNAME)
    envio = GameCoopEnv(run_name=RUNNAME,max_clock=600, msps_requests=MSPS_REQUESTS)
    return envio


# envi = NSEnv(seed=42, runname=RUNNAME, file_name=RUNNAME)
# envi = GameCoopEnv(run_name=RUNNAME,max_clock=1000)

n_procs = 10
# n_procs = 4

if LOAD:
    #Assuming that the model was saved with n_procs
    print(f"trying to load the model from: ./results/{LOADED_MODEL_RUN}/current_model")
    model_path = f"./results/{LOADED_MODEL_RUN}/current_model"
    task = SubprocVecEnv([lambda: Monitor(enviroment_generator()) for _ in range(n_procs)], start_method='fork')
    # model = A2C.load(model_path)
    model = PPO.load(model_path)
    model.set_env(task)  #used to set the enviroment for multiprocessing
    #model.ent_coef = 0.1

else:

    # env = SubprocVecEnv([lambda: Monitor(env) for i in range(n_procs)],start_method='fork')
   
    trainer = RLTrainer(
    env_generator=enviroment_generator,
    run_name=RUNNAME,
    max_steps=MAX_STEPS,
    learning_rate=1e-4,
    gamma=GAMMA,
    n_procs=n_procs,
    policy="MlpPolicy",
    policy_kwargs= dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    batch_size=256

)   
    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # os.system(f"echo Testing @{date_time} >> ./results/{RUNNAME}/configurations.txt")
    append_to_file(f"./results/{RUNNAME}/configurations.txt", f"Testing @{date_time}\n")
    append_to_file(f"./results/{RUNNAME}/configurations.txt", f"{data}\n")

    print("Copying the current enviroment into the results folder.")
    os.system(f"cp EnvCooperation.py ./results/{RUNNAME}/EnvCopy.py")
    
    append_to_file("experiments_log.txt", f"{date_time}\n")
    append_to_file("experiments_log.txt", f"{data}\n")
    append_to_file("experiments_log.txt", f"**************************************************\n")
    model = trainer.train()
    # task = SubprocVecEnv([lambda: Monitor(enviroment_generator()) for _ in range(n_procs)], start_method='fork')
    # model = trainer.load_model(f"./results/{RUNNAME}/current_model")
    # model.set_env(task)
    

