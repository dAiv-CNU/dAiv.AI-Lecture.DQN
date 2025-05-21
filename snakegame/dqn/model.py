from torch import nn
import torch

import random
import time


class SnakeValueNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

        self.value_net = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        return self.value_net(x)


class SnakeModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.device = torch.device("cpu")
        self.input_shape = input_shape
        self.n_actions = n_actions

        self.perception_module = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 입력 크기 계산 (stride=2로 인해 크기가 1/8로 줄어듦)
        conv_out_size = input_shape[0] // 8 * input_shape[1] // 8 * 64

        self.value_net = SnakeValueNet(conv_out_size, n_actions)

    def forward(self, state):
        # 입력 차원 변환: (batch, height, width, channels) -> (batch, channels, height, width)
        x = state.permute(0, 3, 1, 2)

        # 퍼셉션 모듈을 통한 특징 추출
        features = self.perception_module(x)

        # 가치 네트워크를 통한 행동 가치 계산
        action_values = self.value_net(features)

        return action_values

    def select_action(self, state, eps_threshold):
        # 랜덤 시드 새로 설정 (매번 다른 랜덤 행동 보장)
        random_seed = int(time.time() * 1000) % 10000
        random.seed(random_seed)

        if random.random() > eps_threshold:
            with torch.no_grad():
                # 최적 행동 선택
                q_values = self(state)
                return q_values.max(1).indices.view(1, 1)  # get maximum-likely action
        else:  # random action for exploration while early in training
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def to(self, device):
        self.device = device
        return super().to(device)
