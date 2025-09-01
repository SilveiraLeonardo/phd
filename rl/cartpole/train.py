import gymnasium as gym

from buffers.buffers import ReplayBuffer
from models.models import DQN_MLP
from utils.utils import plot_durations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random

# create a simple environment
env = gym.make("CartPole-v1")

# discrete action space (button presses)
print(f'action space: {env.action_space}') # 1 or 2
print(f'sample action: {env.action_space.sample()}')
print('0: push cart to the left; 1: push cart to the right')

# box observation spaece (continuous values)
print(f'observation space: {env.observation_space}') # box with 4 values
print(f'sample observation: {env.observation_space.sample()}')
print('cart_position, cart_velocity, pole_angle, pole_angular_velocity')

print('reward +1 is given for every step taken, including terminating step')

observation, info = env.reset()

# parameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.00
EPS_END = 0.0001
EPS_DECAY = 70000
LR = 1e-3
UPDATE_WEIGHTS_EVERY = 1000
REPLAY_CAPACITY = 100_000
NUM_EPISODES = 3001

# get number of action in the action space
n_actions = env.action_space.n
# get the number of state observations
n_observations = len(observation)

policy_net = DQN_MLP(n_observations, n_actions)
target_net = DQN_MLP(n_observations, n_actions)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR)

replay_buffer = ReplayBuffer(REPLAY_CAPACITY)

epsilon = EPS_START
total_steps = 0
episode_durations = []

for episode in range(NUM_EPISODES):
    state, info = env.reset()
    done = False
    duration = 0
    while not done:
        # select action using e-greedy policy
        rd = random.random()
        if rd < epsilon and total_steps < EPS_DECAY:
            action_selected = env.action_space.sample()
        else:
            with torch.no_grad():
                action_logits = policy_net(torch.FloatTensor(state).unsqueeze(0))
            # next action according to current policy
            action_selected = action_logits.argmax().item()

        # step environment
        next_state, reward, terminated, truncated, info = env.step(action_selected)

        duration += 1
        total_steps += 1
        done = terminated or truncated

        # store transition
        replay_buffer.push(state, action_selected, reward, next_state, done)
        state = next_state

        # learn
        if len(replay_buffer) >= BATCH_SIZE:
            s, a, r, s2, d = replay_buffer.sample(BATCH_SIZE)

            s = torch.FloatTensor(s)
            a = torch.LongTensor(a)
            r = torch.FloatTensor(r)
            s2 = torch.FloatTensor(s2)
            d = torch.FloatTensor(d) # 0s and 1s

            # compute current q-values (what I think the action values are)
            q_vals = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
           
            # what the q-values "really" are - target policy
            next_q = target_net(s2).max(1)[0] # action that maximazes the action-value function of next state
            target = r + (1 - d) * GAMMA * next_q # if episode is done, q-value for the state is only the reward

            # loss and backprop
            # another option here is Huber loss: 
            # means squared error when the error is small, but mean absolute error when it is large
            # makes it more robust to outliers when the estimates are noisy
            loss = F.mse_loss(q_vals, target)

            optimizer.zero_grad()
            loss.backward()

            # optional: clip the gradients for stability
            nn.utils.clip_grad_value_(policy_net.parameters(), clip_value=1.5)
            #nn.utils.clip_grad_norm(policy_net.parameters(), max_norm=1.0)

            optimizer.step()

        episode_durations.append(duration)

        # update target net
        if total_steps % UPDATE_WEIGHTS_EVERY == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # update epsilon
        epsilon = max(EPS_END, EPS_START - total_steps / EPS_DECAY)
    
    # end of episode
    print(f"Episode {episode}, steps {total_steps}, epsilon {epsilon}")

    if episode % 1000 == 0 and episode != 0:
        plot_durations(episode_durations)




