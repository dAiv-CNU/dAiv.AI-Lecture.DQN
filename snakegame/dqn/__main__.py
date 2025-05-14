from .snake import SnakeBoard, config
from .memory import ReplayMemory
from .model import SnakeModel

import pygame

import torch
from torch import optim

from itertools import count
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Pygame 초기화
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode(
        (config.SCREEN_SIZE * config.PIXEL_SIZE, config.SCREEN_SIZE * config.PIXEL_SIZE)
    )
    pygame.display.set_caption("Snake Game with DQN")

    # 환경 및 네트워크 초기화
    env = SnakeBoard()
    n_actions = env.action_space.n
    online_net = SnakeModel((config.SCREEN_SIZE, config.SCREEN_SIZE, 3), n_actions).to(device)
    target_net = SnakeModel((config.SCREEN_SIZE, config.SCREEN_SIZE, 3), n_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(online_net.parameters(), lr=config.LR)
    memory = ReplayMemory(10000, online_net, target_net, optimizer, device, config)

    # 에피소드 반복
    for i_episode in range(config.NUM_EPISODES):
        # 환경 초기화
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0  # 총 보상 초기화

        for steps_done in count():
            # Pygame 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # 행동 선택
            eps_threshold = config.EPS_END + (config.EPS_START - config.EPS_END) * math.exp(-1. * steps_done / config.EPS_DECAY)
            action = online_net.select_action(state, eps_threshold)

            # 환경 단계 진행
            next_state, reward, done, _ = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            next_state_tensor = None if done else torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            # 리플레이 메모리에 저장
            memory.push(state, action, next_state_tensor, reward)

            # 상태 업데이트
            state = next_state_tensor

            # 모델 최적화
            memory.optimize()

            # 게임 렌더링
            env.render(screen)

            if done:
                break

        # 타겟 네트워크 소프트 업데이트
        with torch.no_grad():
            for key in online_net.state_dict():
                target_net.state_dict()[key].data.copy_(
                    config.TAU * online_net.state_dict()[key].data + (1.0 - config.TAU) * target_net.state_dict()[key].data
                )

        # 에피소드 결과 출력
        print(f"Episode {i_episode + 1}: Total Reward = {total_reward}, Score = {env.score}")

    # Pygame 종료
    pygame.quit()


if __name__ == '__main__':
    main()
