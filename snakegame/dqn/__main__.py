from .snake import SnakeBoard, config
from .memory import ReplayMemory
from .model import SnakeModel

import pygame

import torch
from torch import optim

from itertools import count
import random
import math
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"INFO: Using device - {device}")


def main():
    # Pygame 초기화
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode(
        (config.SCREEN_SIZE * config.PIXEL_SIZE, config.SCREEN_SIZE * config.PIXEL_SIZE)
    )
    pygame.display.set_caption("Snake Game with DQN")

    # 환경 및 네트워크 초기화
    initial_speed_multiplier = 100.0  # 초기 속도 증가 배수 (매우 빠르게)
    env = SnakeBoard(speed_multiplier=initial_speed_multiplier)
    n_actions = env.action_space.n
    online_net = SnakeModel((config.SCREEN_SIZE, config.SCREEN_SIZE, 3), n_actions).to(device)
    target_net = SnakeModel((config.SCREEN_SIZE, config.SCREEN_SIZE, 3), n_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    # 학습률 조정
    optimizer = optim.AdamW(online_net.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    memory = ReplayMemory(20000, online_net, target_net, optimizer, device, config)

    # 초기화
    learning_starts = 5000
    target_update_freq = 100

    # 처음 1000번의 에피소드는 빠른 속도로 실행
    fast_mode_episodes = 1000  # 빠른 모드에서 실행할 에피소드 수

    # 전체 에피소드 루프
    for i_episode in range(config.NUM_EPISODES):
        # 1000번 에피소드 이후에는 일반 속도로 전환
        if i_episode == fast_mode_episodes:
            print(f"\n[INFO] 빠른 학습 모드 종료 ({fast_mode_episodes}개 에피소드 완료)")
            # 일반 속도로 환경 재설정
            env.speed_multiplier = 1.0
            print("[INFO] 일반 속도 모드로 전환\n")

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
            if steps_done < learning_starts:
                random.seed(time.time_ns() % 100000 + steps_done)
                action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
                eps_threshold = 1.0
            else:
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
            if steps_done >= learning_starts:
                memory.optimize()

            # 타겟 네트워크 업데이트
            if steps_done >= learning_starts and steps_done % target_update_freq == 0:
                target_net.load_state_dict(online_net.state_dict())

            # 게임 렌더링
            env.render(screen)

            if done:
                break

        # 빠른 모드 진행 상황 출력 (빠른 모드에서는 10개 에피소드마다 출력)
        if i_episode < fast_mode_episodes:
            if (i_episode + 1) % 10 == 0:  # 10개 에피소드마다 출력
                print(f"빠른 학습: {i_episode + 1}/{fast_mode_episodes} 에피소드 완료")
        else:
            # 일반 모드에서는 매 에피소드마다 출력
            print(f"에피소드 {i_episode + 1}: 총 보상 = {total_reward}, 점수 = {env.score}")

    # Pygame 종료
    pygame.quit()


if __name__ == '__main__':
    main()
