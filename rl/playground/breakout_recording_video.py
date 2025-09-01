import gymnasium as gym
import ale_py

# initialialize environmnet
gym.register_envs(ale_py)

# rgb_array return the rgb array from env.render()
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

# wrap it under RecordVideo wrapper
env = gym.wrappers.RecordVideo(
    env,
    episode_trigger=lambda num: num % 2 == 0, # render every other episode
    video_folder="saved_videos",
    name_prefix="video-",
)


# render_mode="human" will automatically create a window running at 60 fps

# discrete action space (button presses)
print(f'action space: {env.action_space}')
print(f'sample action: {env.action_space.sample()}')

# box observation spaece (continuous values)
print(f'observation space: {env.observation_space}') # box with 4 values

for episode in range(10):

    observation, info = env.reset(seed=42)
    episode_over = False

    while not episode_over:
        # sample by the current policy
        action = env.action_space.sample()

        # sample from the environment given current state and action
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

env.close()
