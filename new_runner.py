import os
import subprocess

from tqdm import tqdm
import numpy as np
import sounddevice as sd

def beep(frequency=1000, duration=0.3, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    sd.play(wave, samplerate=sample_rate)
    sd.wait()
# for num_msps in range(1,10,2):

# runname_extra_name = "check_rew4_oldintermediate_msps_5_changedroomcost"
# runname_extra_name = "non_cooper_msps_5"
num_msps = 2
runname_extra_name = "testing"
# for num_requests in tqdm(range(10, 100, 20)):
for num_requests in tqdm(range(30, 40, 20)):
# for num_requests in tqdm(range(1000, 1020, 20)):
    # for num_requests in tqdm(range(10,100,20)):
    runname = f"msps_{num_msps}_requests_{num_requests}_gamma_0.97_{runname_extra_name}"
    # num_episodes_required = 500_000
    num_episodes_required = 100_000
    num_training_steps = num_episodes_required * num_requests
    gamma = 0.97
    print("running runname: ", runname, "with num_training_steps: ", num_training_steps)
    subprocess.run(
        ["python", "experiment_runner.py", "--runname", runname, "--gamma", f"{gamma}", "--ls", str(num_training_steps),
         "--msg", f"trying {runname}", "--msps_requests", str(num_requests)]
    )
beep()

# changed the intermediate reward to the old one. to see if the change from ppo to a2c is the reason for the bad performance.
