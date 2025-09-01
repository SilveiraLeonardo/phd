import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

# create a simple environment
env = gym.make("CarRacing-v3")

print(env.observation_space.shape)

# wrap it to flatten the observation into a 1D array
wrapped_env = FlattenObservation(env)
print(wrapped_env.observation_space.shape)
