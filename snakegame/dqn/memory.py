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

        # 현재 상태-행동 가치 계산
        state_action_values = self.online_net(state_batch).gather(1, action_batch)

        # 다음 상태 가치 계산 (Double DQN 방식)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            if non_final_mask.any():
                # 일반 DQN: 타겟 네트워크로 최적 행동 가치 계산
                # 초기에는 일반 DQN이 더 안정적일 수 있음
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # 기대 상태-행동 가치 계산
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # 벨만 오차 계산
        bellman_error = expected_state_action_values.unsqueeze(1) - state_action_values

        # 벨만 오차 클리핑 (-1, 1)
        clipped_bellman_error = bellman_error.clamp(-1, 1)

        # 역방향 그래디언트 계산
        d_error = clipped_bellman_error * -1.0

        # 그래디언트 업데이트
        state_action_values.backward(d_error)

        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)

        # 옵티마이저 스텝
        self.optimizer.step()
