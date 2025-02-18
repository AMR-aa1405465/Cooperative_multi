import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import SAC
from stable_baselines3.common.envs import DummyVecEnv
from gym import Env, spaces


# Step 1: Define the Autoencoder for Latent Action Space
class ActionAutoencoder(nn.Module):
    def __init__(self, input_dim=10, latent_dim=3):
        super(ActionAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


# Initialize the autoencoder
latent_dim = 3  # Compress 10D action space into 3D
autoencoder = ActionAutoencoder(input_dim=10, latent_dim=latent_dim)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
loss_function = nn.MSELoss()

# Dummy dataset for training the autoencoder (simulate valid action sequences)
num_samples = 10000
action_data = torch.randint(0, 64, (num_samples, 10), dtype=torch.float32) / 63.0  # Normalize to [0,1]


def train_autoencoder(autoencoder, action_data, epochs=20, batch_size=128):
    autoencoder.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, num_samples, batch_size):
            batch = action_data[i:i + batch_size]
            optimizer.zero_grad()
            latent, reconstructed = autoencoder(batch)
            loss = loss_function(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / (num_samples // batch_size):.4f}")
    print("Autoencoder training completed.")


# Train the autoencoder
train_autoencoder(autoencoder, action_data)


# Step 2: Define Custom Gym Environment with Latent Action Space
class LatentActionEnv(Env):
    def __init__(self, autoencoder, latent_dim=3):
        super(LatentActionEnv, self).__init__()
        self.autoencoder = autoencoder
        self.latent_dim = latent_dim
        self.action_space = spaces.Box(low=-1, high=1, shape=(latent_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.state = np.random.rand(10)
        print("Environment initialized.")

    def step(self, action):
        print(f"Received action: {action}")
        # Decode action from latent space
        with torch.no_grad():
            action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
            decoded_action = autoencoder.decoder(action_tensor).squeeze().numpy()
            decoded_action = (decoded_action * 63).astype(int)  # Scale back to [0, 63]
            decoded_action = np.clip(decoded_action, 0, 63)  # Ensure valid range
        print(f"Decoded action: {decoded_action}")

        # Simulate some reward function (can be adapted based on your needs)
        reward = -np.sum(np.abs(decoded_action - 32))  # Example: closer to 32 is better
        print(f"Calculated reward: {reward}")
        done = False  # Define stopping condition if needed
        self.state = np.random.rand(10)  # Update state randomly
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.random.rand(10)
        print("Environment reset.")
        return self.state

    def render(self):
        pass


# Step 3: Train SAC Agent on Latent Action Space
env = DummyVecEnv([lambda: LatentActionEnv(autoencoder, latent_dim=latent_dim)])

# Train SAC with Latent Action Space
print("Starting SAC training...")
model = SAC("MlpPolicy", env, verbose=1, learning_rate=0.001, buffer_size=100000)
model.learn(total_timesteps=50000)
print("SAC training completed.")

# Test the trained agent
done = False
obs = env.reset()
while not done:
    action, _ = model.predict(obs)
    print(f"Agent selected action: {action}")
    obs, reward, done, _ = env.step(action)
    print(f"Received reward: {reward}")