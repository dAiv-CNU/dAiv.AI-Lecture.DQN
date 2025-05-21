from torch import nn
import torch
import torch.nn.functional as F
import random
import time


# class SnakePolicyNet(nn.Module):
#     def __init__(self, input_shape, n_actions):
#         super().__init__()
#         self.input_shape = input_shape
#         self.n_actions = n_actions
#
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#
#         self.relu = nn.ReLU()
#
#         self.fc1 = nn.Linear(2560*40, 512)
#         self.fc2 = nn.Linear(512, n_actions)
#
#     def forward(self, x):
#         # Reshape input: (batch, height, width, channels) -> (batch, channels, height, width)
#         x = x.permute(0, 3, 1, 2)
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = self.relu(self.fc1(torch.flatten(x, 1)))
#         return self.fc2(x)


class SnakeValueNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

        # 퍼셉션 모듈에서 나온 특징 맵의 평탄화 후 처리
        # 입력 크기는 perception_module의 출력 크기에 따라 계산됨
        feature_size = 64 * input_shape[0] * input_shape[1]  # 64 채널 * 높이 * 너비

        self.value_net = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        # 특징 맵을 평탄화하여 완전 연결 레이어에 전달
        x = torch.flatten(x, 1)
        return self.value_net(x)


class SnakeModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.device = torch.device("cpu")
        self.input_shape = input_shape
        self.n_actions = n_actions

        self.perception_module = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.value_net = SnakeValueNet(input_shape, n_actions)

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
