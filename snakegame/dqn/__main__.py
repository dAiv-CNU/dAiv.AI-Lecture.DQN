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
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"INFO: Using device - {device}")

if not os.path.isdir(config.CHECKPOINT_PATH):
    os.makedirs(config.CHECKPOINT_PATH)


def main():
    # Pygame 초기화
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode(
        (config.SCREEN_SIZE * config.PIXEL_SIZE, config.SCREEN_SIZE * config.PIXEL_SIZE)
    )
    pygame.display.set_caption("Snake Game with DQN")

    # 환경 및 네트워크 초기화
    initial_speed_multiplier = 10.0
    env = SnakeBoard(speed_multiplier=initial_speed_multiplier)
    n_actions = env.action_space.n
    online_net = SnakeModel((config.SCREEN_SIZE, config.SCREEN_SIZE, 3), n_actions).to(device)
    target_net = SnakeModel((config.SCREEN_SIZE, config.SCREEN_SIZE, 3), n_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    # 학습률 조정
    optimizer = optim.AdamW(online_net.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    memory = ReplayMemory(20000, online_net, target_net, optimizer, device, config)

    # 체크포인트 로드
    if os.path.exists(os.path.join(config.CHECKPOINT_PATH, "checkpoint.pth")):
        checkpoint = torch.load(os.path.join(config.CHECKPOINT_PATH, "checkpoint.pth"), map_location=device)
        online_net.load_state_dict(checkpoint['model_state_dict'])
        target_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.AdamW(online_net.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("[INFO] 체크포인트 로드 완료")

    # 전체 에피소드 루프
    for i_episode in range(config.NUM_EPISODES):
        # fast 에피소드 이후에는 일반 속도로 전환
        if i_episode == config.FAST_EPISODES:
            print(f"\n[INFO] 빠른 학습 모드 종료 ({config.FAST_EPISODES}개 에피소드 완료)")
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
            if steps_done < config.LEARNING_STARTS:
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
            if steps_done >= config.LEARNING_STARTS:
                memory.optimize()

            # 게임 렌더링
            env.render(screen)

            if done:
                break

        # 타겟 네트워크 소프트 업데이트
        if steps_done >= config.LEARNING_STARTS:
            with torch.no_grad():
                for key in online_net.state_dict():
                    target_net.state_dict()[key].data.copy_(
                        config.TAU * online_net.state_dict()[key].data + (1.0 - config.TAU) * target_net.state_dict()[key].data
                    )

        # 빠른 모드 진행 상황 출력 (빠른 모드에서는 10개 에피소드마다 출력)
        if i_episode < config.FAST_EPISODES:
            if (i_episode + 1) % 10 == 0:  # 10개 에피소드마다 출력
                print(f"빠른 학습: {i_episode + 1}/{config.FAST_EPISODES} 에피소드 완료")
        else:
            # 일반 모드에서는 매 에피소드마다 출력
            print(f"에피소드 {i_episode + 1}: 총 보상 = {total_reward}, 점수 = {env.score}")
            
            # 체크포인트 저장
            if (i_episode + 1) % config.CHECKPOINT_INTERVAL == 0:
                checkpoint = {
                    'model_state_dict': online_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode': i_episode + 1
                }
                torch.save(checkpoint, os.path.join(config.CHECKPOINT_PATH, "checkpoint.pth"))
                print(f"[INFO] 체크포인트 저장 완료 (에피소드 {i_episode + 1})")

    # Pygame 종료
    pygame.quit()


if __name__ == '__main__':
    main()
