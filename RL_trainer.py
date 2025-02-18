import glob
import os
from typing import Callable, Type, Union, Dict, Any
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import pandas as pd 
import matplotlib.pyplot as plt
from stable_baselines3.common.logger import configure
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

import signal
# from stable_baselines3.common.utils import linear_schedule

class RLTrainer:
    def __init__(
        self,
        env_generator: Callable,
        run_name: str,
        n_procs: int = 4,
        max_steps: int = 1_000_000,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        # learning_rate: float = 1e-4,
        seed: int = 1234,
        batch_size: int = 64,
        policy: Union[str, Type] = "MlpPolicy",
        policy_kwargs: Dict[str, Any] = None,
        load_model: bool = False,
        model_path: str = None,
        enable_wandb: bool = False
    ):
        """
        Initialize RL training setup.
        
        Args:
            env_generator: Function that creates the environment
            run_name: Name of the training run
            n_procs: Number of parallel environments
            max_steps: Total timesteps to train
            gamma: Discount factor
            learning_rate: Learning rate
            seed: Random seed
            policy: Custom policy network (optional)
            policy_kwargs: Additional arguments for the policy network (optional)
        """
        self.env_generator = env_generator
        self.run_name = run_name
        self.n_procs = n_procs
        self.max_steps = max_steps
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.seed = seed
        self.policy = policy
        self.policy_kwargs = policy_kwargs
        self.batch_size = batch_size
        self.enable_wandb = enable_wandb
        
        # Create directories
        self.setup_directories()
        
        # Setup environment and model
        self.env = self.create_env()
        if load_model:
            self.model = self.load_model(model_path)
        else:
            self.model = self.setup_model()
        self.callbacks = self.setup_callbacks()
        

       # Flag for graceful interruption
        self.interrupted = False
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)
        # self.train()
    def get_parameters_str(self):
        return f"run_name: {self.run_name}, n_procs: {self.n_procs}, max_steps: {self.max_steps}, gamma: {self.gamma}, learning_rate: {self.learning_rate}, seed: {self.seed}, batch_size: {self.batch_size}, policy: {self.policy}, policy_kwargs: {self.policy_kwargs}"
    def load_model(self, path: str):
        """Load a saved model and set up for continued training."""
        # Configure new logger for continued training
        new_logger = configure(f"./results/{self.run_name}/", ["csv"])
        
        # Load the model
        model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}",model.device)
        
        # Set the new logger
        model.set_logger(new_logger)
        
        return model
    def handle_interrupt(self, signum, frame):
        """Handle keyboard interruption (Ctrl+C)"""
        print("\nInterrupt received! Saving model before stopping...")
        self.interrupted = True
        current_model_path = f"{self.results_dir}/interrupted_current_model"
        self.save_model(path=current_model_path)
        print("Model saved successfully. You can safely exit now.")

    def setup_directories(self):
        """Create necessary directories for saving results and logs."""
        self.results_dir = f"./results/{self.run_name}"
        # os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs("./logs", exist_ok=True)

    def create_env(self):
        """Create vectorized environment."""
        def make_env(rank):
            def _init():
                env = self.env_generator()
                #env.seed(self.seed + rank)
                return Monitor(env)
            return _init

        return SubprocVecEnv(
            [make_env(i) for i in range(self.n_procs)],
            start_method='fork'
        )

    def setup_model(self):
        """Initialize PPO model."""
        new_logger = configure(f"./results/{self.run_name}/", ["csv"])
        mm = PPO(
            self.policy,
            self.env,
            verbose=1,
            gamma=self.gamma,
            n_steps=2048,   # Increased for better training stability
            batch_size=64,  # Smaller batch size for better generalization
            n_epochs=10,    # Multiple epochs per update for better sample efficiency
            ent_coef=0.01,  # Encourage exploration
            learning_rate=3e-4,
            clip_range=0.2,  # Standard PPO :lipping
            gae_lambda=0.95, # GAE factor for advantage estimation
            policy_kwargs=self.policy_kwargs,
            normalize_advantage=True,  # Helps with training stability
            #tensorboard_log=f"./logs/{self.run_name}"  # Enable tensorboard logging
        )
        # mm = SAC(
        #     self.policy,
        #     self.env,
        #     verbose=1,
        #     gamma=self.gamma,
        #     batch_size=64,
        #     gradient_steps=10,
        #     policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
        # )
        # mm =  PPO(
        #     self.policy,
        #     self.env,
        #     verbose=1,
        #     gamma=self.gamma,
        #     n_steps=450,#400,
        #     seed=self.seed,
        #     learning_rate=self.learning_rate,
        #     policy_kwargs=self.policy_kwargs,
        #     #batch_size=512,#self.batch_size,
        #     # device=''
        #     # tensorboard_log="./logs"
        # )
        # mm = A2C(
        #     self.policy,
        #     self.env,
        #     # n_steps=50, # before it was originally not SET.
        #     # n_steps=1000,
        #     # n_steps=500,
        #     n_steps=100,
        #     verbose=1,
        #     gamma=self.gamma,
        #     learning_rate=self.learning_rate,
        #     policy_kwargs=self.policy_kwargs,
        #     # ent_coef=0.002,
        #     # ent_coef=linear_schedule(0.05), # initially it wasn't there.
        #     # batch_size=self.batch_size,
        # )
        mm.set_logger(new_logger)
        return mm

    def setup_callbacks(self):
        # weights and biases
        callbacks = []
        if self.enable_wandb:
            wandb.init(project="cooperative-game",sync_tensorboard=True)
            wandb_callback = WandbCallback(
                gradient_save_freq=100,
                # model_save_freq=50000,
                model_save_path=f"{self.results_dir}/wandb/",
                verbose=2
            )

        """Setup training callbacks."""
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=f"{self.results_dir}/checkpoints",
            name_prefix="model"
        )

        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=f"{self.results_dir}/best_model",
            eval_freq=10000,
            deterministic=True,
            render=False
        )
        callbacks.append(checkpoint_callback)
        callbacks.append(eval_callback)
        if self.enable_wandb:
            callbacks.append(wandb_callback)
        return callbacks

    def save_model(self,path:str):
        """Save the current model."""
        #model_path = f"{self.results_dir}/current_model"
        self.model.save(path)
        print(f"Model saved to {path}")

    def plot_training_progress(self, save_fig: bool = True, show_fig: bool = True):
        """
        Plot episode rewards and lengths from monitor logs.
        """
        try:
            # Read monitor files
            monitor_files = glob.glob(os.path.join(self.results_dir, "*.monitor.csv"))
            if not monitor_files:
                print("No monitor files found.")
                return

            dfs = []
            for file in monitor_files:
                # Skip first two lines (metadata)
                df = pd.read_csv(file, skiprows=2)
                dfs.append(df)

            data = pd.concat(dfs)
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            
            # Plot episode rewards
            plt.subplot(1, 2, 1)
            plt.plot(data.index, data['r'], label='Episode Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Episode Rewards over Training')
            plt.grid(True)
            
            # Plot episode lengths
            plt.subplot(1, 2, 2)
            plt.plot(data.index, data['l'], label='Episode Length', color='orange')
            plt.xlabel('Episode')
            plt.ylabel('Length')
            plt.title('Episode Lengths over Training')
            plt.grid(True)

            plt.tight_layout()

            if save_fig:
                plt.savefig(f"{self.results_dir}/training_progress.png")
                print(f"Training progress plots saved to {self.results_dir}/training_progress.png")

            if show_fig:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            print(f"Error plotting training progress: {str(e)}")

    def plot_training_progress(self, save_fig: bool = True, show_fig: bool = True):
        """
        Plot episode rewards and lengths from monitor logs.
        """
        try:
            # Read monitor files
            monitor_files = glob.glob(os.path.join(self.results_dir, "*.monitor.csv"))
            if not monitor_files:
                print("No monitor files found.")
                return

            dfs = []
            for file in monitor_files:
                # Skip first two lines (metadata)
                df = pd.read_csv(file, skiprows=2)
                dfs.append(df)

            data = pd.concat(dfs)
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            
            # Plot episode rewards
            plt.subplot(1, 2, 1)
            plt.plot(data.index, data['r'], label='Episode Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Episode Rewards over Training')
            plt.grid(True)
            
            # Plot episode lengths
            plt.subplot(1, 2, 2)
            plt.plot(data.index, data['l'], label='Episode Length', color='orange')
            plt.xlabel('Episode')
            plt.ylabel('Length')
            plt.title('Episode Lengths over Training')
            plt.grid(True)

            plt.tight_layout()

            if save_fig:
                plt.savefig(f"{self.results_dir}/training_progress.png")
                print(f"Training progress plots saved to {self.results_dir}/training_progress.png")

            if show_fig:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            print(f"Error plotting training progress: {str(e)}")

    def train(self):
        """Start the training process."""
        try:
            print(f"Starting training for {self.max_steps} timesteps...")
            self.model.learn(
                total_timesteps=self.max_steps,
                callback=self.callbacks,
                progress_bar=True,
            )
            

            # Save final model
            final_model_path = f"{self.results_dir}/final_model"
            self.model.save(final_model_path)
            print(f"Training completed. Final model saved to {final_model_path}")
            
            return self.model
            
        except Exception as e:
            
            final_model_path = f"{self.results_dir}/temp_final_model"
            self.model.save(final_model_path)
            print(f"Training failed with error: {str(e)}")
            raise
        finally:
            self.env.close()

    # def load_model(self, path: str):
    #     """Load a saved model."""
    #     return PPO.load(path, env=self.env)

# def env_gen():
#     envio = GameCoopEnv(runname="test",max_cl)
#     return envio


# if __name__ == "__main__":
#     from EnvCooperation import GameCoopEnv
#     env = GameCoopEnv("test3",m)
#     trainer = RLTrainer(env_generator=env_gen, run_name="test", n_procs=4, max_steps=100000)
#     new_model = trainer.load_model("./results/smaller_budget/best_model/best_model.zip")
    
#     done = False
#     obs, info = env.reset()
#     while not done:
#         action, _states = new_model.predict(obs, deterministic=True)
#         obs, reward, done, trunc, info = env.step(action)
#         print(f"action: {action}, reward: {reward}, done: {done}, trunc: {trunc}, info: {info}")
#         # env.render()   


   # trainer.train()