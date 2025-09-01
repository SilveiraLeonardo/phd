import numpy as np
import random

# stores tuples (state, action, reward, next_state, done)
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, s, a, r, s2, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (s, a, r, s2, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        # instead of a list of examples, turn them into an array of each of the things:
        # array of states, another of actions, another of rewards, etc.
        s, a, r, s2, d = map(np.stack, zip(*batch))

        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)



