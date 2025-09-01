import gymnasium as gym

# initialialize environmnet
env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset(seed=42)

for _ in range(1000):
    # sample by the current policy
    action = env.action_space.sample()

    # sample from the environment given current state and action
    observation, reward, terminated, truncated, info = env.step(action)

    # if episode ended, reset it
    if terminated or truncated:
        observation, info = env.reset()

env.close()
