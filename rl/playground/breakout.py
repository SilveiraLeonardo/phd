import gymnasium as gym
import ale_py

# initialialize environmnet
gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode="human")

# render_mode="human" will automatically create a window running at 60 fps

# discrete action space (button presses)
print(f'action space: {env.action_space}')
print(f'sample action: {env.action_space.sample()}')

# box observation spaece (continuous values)
print(f'observation space: {env.observation_space}') # box with 4 values

print(b)

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
