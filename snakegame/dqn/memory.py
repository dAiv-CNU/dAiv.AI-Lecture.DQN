from collections import namedtuple, deque

from torch.nn import functional as F
import torch

import random


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory:
    trainsition_type = Transition

    def __init__(self, capacity, online_net, target_net, optimizer, device, config):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity
        self.online_net = online_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.device = device
        self.gamma = config.GAMMA
        self.batch_size = config.BATCH_SIZE

    def push(self, *args):
        self.memory.append(self.trainsition_type(*args))

    @property
    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)

    def optimize(self):
        self.optimizer.zero_grad()

        if len(self) < self.batch_size:
            return
        transitions = self.sample
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device, dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.online_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        loss.backward()
        self.optimizer.step()
