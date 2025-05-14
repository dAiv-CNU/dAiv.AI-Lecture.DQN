from itertools import count
import os
import time

import pygame
import torch
from torch import optim

import config
import base
import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main_dqn():
    # Pygame 초기화
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode(
        (config.SCREEN_SIZE * config.PIXEL_SIZE, config.SCREEN_SIZE * config.PIXEL_SIZE)
    )
    pygame.display.set_caption('Snake Game with DQN')

    # 환경 및 네트워크 초기화
    env = base.SnakeEnv()
    n_actions = env.action_space.n

    # 네트워크 초기화
    policy_net = model.DQN((config.SCREEN_SIZE, config.SCREEN_SIZE), n_actions).to(device)
    target_net = model.DQN((config.SCREEN_SIZE, config.SCREEN_SIZE), n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # 옵티마이저: Adam에 가중치 감소 추가
    optimizer = optim.Adam(policy_net.parameters(), lr=model.LR, weight_decay=1e-5)
    # 학습률 스케줄러 추가
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    memory = model.ReplayMemory(50000)  # 메모리 크기 증가

    # 모델 저장 디렉토리 생성
    os.makedirs('models', exist_ok=True)

    # 간단한 성과 추적 (저장용)
    episode_rewards = []
    episode_scores = []
    episode_losses = []
    best_score = 0

    print("학습을 시작합니다...")
    start_time = time.time()

    # 에피소드 반복
    for i_episode in range(model.NUM_EPISODES):
        # 환경 초기화
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        episode_loss = 0
        loss_count = 0

        for t in count():
            # Pygame 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # 학습 종료 전 모델만 저장
                    torch.save(policy_net.state_dict(), 'models/dqn_final.pth')
                    pygame.quit()
                    return

            # 행동 선택 (인스턴스 메서드로 변경됨)
            action = policy_net.select_action(state)

            # 환경 단계 진행
            next_state, reward, done, _, _ = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            next_state_tensor = None if done else torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            # 리플레이 메모리에 저장
            memory.push(state, action, next_state_tensor, reward)

            # 상태 업데이트
            state = next_state_tensor

            # 모델 최적화 (함수 반환값 받기)
            loss = model.optimize_model(memory, policy_net, target_net, optimizer)
            if loss is not None:
                episode_loss += loss
                loss_count += 1

            # 매 프레임마다 게임 렌더링 (성능에 영향이 있을 수 있음)
            env.render(screen)

            # 게임 속도 제어 (필요한 경우 주석 해제)
            # pygame.time.delay(30)

            if done:
                break

        # 타겟 네트워크 소프트 업데이트
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * model.TAU + target_net_state_dict[key] * (1 - model.TAU)
        target_net.load_state_dict(target_net_state_dict)

        # 학습률 업데이트
        scheduler.step()

        # 최소한의 기록 유지 (그래프 없이)
        episode_rewards.append(total_reward)
        episode_scores.append(env.score)
        episode_losses.append(episode_loss / max(1, loss_count) if loss_count > 0 else 0)

        # 최고 점수 모델 저장
        if env.score > best_score:
            best_score = env.score
            torch.save(policy_net.state_dict(), f'models/dqn_best_score_{best_score}.pth')

        # 주기적 체크포인트 저장
        if (i_episode + 1) % 500 == 0:
            torch.save(policy_net.state_dict(), f'models/dqn_checkpoint_{i_episode+1}.pth')

    # 최종 모델 저장
    torch.save(policy_net.state_dict(), 'models/dqn_final.pth')

    # Pygame 종료
    pygame.quit()

if __name__ == "__main__":
    main_dqn()