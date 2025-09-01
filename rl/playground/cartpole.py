import gymnasium as gym

# create a simple environment
env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset(seed=42)

# discrete action space (button presses)
print(f'action space: {env.action_space}')
print(f'sample action: {env.action_space.sample()}')

# box observation spaece (continuous values)
print(f'observation space: {env.observation_space}') # box with 4 values
print(f'sample observation: {env.observation_space.sample()}')

print(b)

print(f'initial observation: {observation}')
# cart_position, cart_velocity, pole_angle, pole_angular_velocity
total_reward = 0

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    # reward +1 for each step the pole stays upright
    # terminated: True if pole falls too far (agent failed)
    # truncated: True if hit time limit (500 steps)

    if terminated or truncated:
        observation, info = env.reset()


env.close()
