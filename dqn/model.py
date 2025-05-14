import random
import math
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 하이퍼파라미터 최적화
BATCH_SIZE = 128  # 더 큰 배치 사이즈로 안정성 향상
GAMMA = 0.99  # 할인율
EPS_START = 1.0  # 초기 탐험률
EPS_END = 0.05  # 최종 탐험률
EPS_DECAY = 1000  # 탐험률 감소 속도
TAU = 0.005  # 타겟 네트워크 소프트 업데이트 비율
LR = 0.0005  # 학습률 감소
NUM_EPISODES = 10000  # 에피소드 수

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        # 입력 채널 수정: 3채널(RGB) 대신 1채널(특징맵)로 변경
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 배치 정규화 추가
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 배치 정규화 추가
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  # 배치 정규화 추가

        # 완전 연결 레이어 개선
        self.fc1 = nn.Linear(2560*40, 256)
        self.fc2 = nn.Linear(256, n_actions)

        self.n_actions = n_actions
        self.steps_done = 0

    def _get_conv_output_size(self, shape):
        # 컨볼루션 레이어의 출력 크기를 동적으로 계산
        input = torch.zeros(1, 1, shape[0], shape[1])
        output = self.bn3(self.conv3(self.bn2(self.conv2(self.bn1(self.conv1(input))))))
        return int(np.prod(output.size()))

    def forward(self, x):
        # x 형태: [batch, height, width, channels]
        # 채널 차원 변경 (특징맵으로 변환)
        x = x.permute(0, 3, 1, 2)  # [batch, channels, height, width]

        # 3채널(RGB)을 1채널(특징맵)으로 변환 (평균)
        x = x.mean(dim=1, keepdim=True)

        # 컨볼루션 레이어
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # 완전 연결 레이어
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def select_action(self, state, training=True):
        # 클래스 메서드를 인스턴스 메서드로 변경
        if training:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
            self.steps_done += 1

            if random.random() > eps_threshold:
                with torch.no_grad():
                    return self(state).max(1).indices.view(1, 1)
            else:
                return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
        else:
            # 평가 모드일 때는 항상 최적 행동 선택
            with torch.no_grad():
                return self(state).max(1).indices.view(1, 1)

def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # 종료 상태가 아닌 마스크 생성
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a) 계산 - 네트워크에서 선택한 행동의 Q값만 추출
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 다음 상태의 최대 예상 Q값 계산: V(s_{t+1}) = max_a Q(s_{t+1}, a)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    # Double DQN 구현: 과대평가 방지를 위해 행동 선택과 평가를 분리
    with torch.no_grad():
        # 정책 네트워크로 행동 선택
        next_action_indices = policy_net(non_final_next_states).max(1).indices.unsqueeze(1)
        # 타겟 네트워크로 행동 평가
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_action_indices).squeeze(1)

    # 기대 Q 값 계산
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber 손실 계산
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()

    # 경사 클리핑: 안정적인 학습을 위해
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 100)

    optimizer.step()

    return loss.item()  # 손실값 반환