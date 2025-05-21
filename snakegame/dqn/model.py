from random import random, randrange

from torch import nn
import torch


class SnakePolicyNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(2560*40, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        # Reshape input: (batch, height, width, channels) -> (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.fc1(torch.flatten(x, 1)))
        return self.fc2(x)


class SnakeValueNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

    def forward(self, x):
        return self.value_net(x)


class SnakeModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.device = torch.device("cpu")
        self.input_shape = input_shape
        self.n_actions = n_actions

        self.policy_net = SnakePolicyNet(input_shape, n_actions)
        self.value_net = SnakeValueNet(input_shape, n_actions)

    def forward(self, state):
        action = self.policy_net(state)
        return action

    def select_action(self, state, eps_threshold):
        if random() > eps_threshold:
            with torch.no_grad():
                return self(state).max(1).indices.view(1, 1)  # get maximum-likely action
        else:  # random action for exploration while early in training
            return torch.tensor([[randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def to(self, device):
        self.device = device
        return super().to(device)
