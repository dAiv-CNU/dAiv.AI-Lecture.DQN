from ..normal.snake import DIRECTIONS, config

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class SnakeBoard(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, speed_multiplier=1.0):
        super().__init__()
        self.action_space = spaces.Discrete(4)  # 4개의 방향 (UP, RIGHT, DOWN, LEFT)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(config.SCREEN_SIZE, config.SCREEN_SIZE, 3), dtype=np.uint8
        )
        # 게임 속도 조절을 위한 변수 추가
        self.speed_multiplier = speed_multiplier
        self.reset()

    def reset(self, seed=None, options=None):
        # 랜덤 시드 설정
        if seed is not None:
            super().reset(seed=seed)
            random.seed(seed)
            np.random.seed(seed)
        else:
            super().reset(seed=random.randint(0, 10000))

        center_x, center_y = config.SCREEN_SIZE // 2, config.SCREEN_SIZE // 2

        # 초기 방향 랜덤화 (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
        self.direction = random.randrange(4)

        # 방향에 따라 다른 초기 위치 설정
        if self.direction == 0:  # UP
            self.snake = np.array([
                [center_x, center_y],
                [center_x, center_y + 1],
                [center_x, center_y + 2],
                [center_x, center_y + 3]
            ])
        elif self.direction == 1:  # RIGHT
            self.snake = np.array([
                [center_x, center_y],
                [center_x - 1, center_y],
                [center_x - 2, center_y],
                [center_x - 3, center_y]
            ])
        elif self.direction == 2:  # DOWN
            self.snake = np.array([
                [center_x, center_y],
                [center_x, center_y - 1],
                [center_x, center_y - 2],
                [center_x, center_y - 3]
            ])
        else:  # LEFT
            self.snake = np.array([
                [center_x, center_y],
                [center_x + 1, center_y],
                [center_x + 2, center_y],
                [center_x + 3, center_y]
            ])

        self.score = 0
        self.done = False
        self.place_fruit()
        return self.get_observation(), {}

    def place_fruit(self):
        while True:
            x = random.randint(0, config.SCREEN_SIZE - 1)
            y = random.randint(0, config.SCREEN_SIZE - 1)
            if list([x, y]) not in self.snake.tolist():
                break
        self.fruit = np.array([x, y])

    def get_observation(self):
        obs = np.zeros((config.SCREEN_SIZE, config.SCREEN_SIZE, 3), dtype=np.uint8)
        for bit in self.snake:
            obs[bit[1], bit[0], 0] = 1  # 뱀
        obs[self.fruit[1], self.fruit[0], 1] = 1  # 과일
        return obs

    def step(self, action):
        if self.done:
            return self.get_observation(), 0, self.done, {}

        # 반대 방향으로 이동하는 것 방지 (0<->2, 1<->3)
        opposite_direction = (self.direction + 2) % 4
        if action == opposite_direction and len(self.snake) > 1:
            # 반대 방향으로 이동하려고 하면 현재 방향 유지
            action = self.direction

        self.direction = action
        old_head = self.snake[0]
        movement = DIRECTIONS[self.direction]
        new_head = old_head + movement

        # 이전 과일과의 거리 계산
        old_dist_to_fruit = np.linalg.norm(old_head - self.fruit)

        # 충돌 검사
        if (new_head[0] < 0 or new_head[0] >= config.SCREEN_SIZE or
                new_head[1] < 0 or new_head[1] >= config.SCREEN_SIZE or
                new_head.tolist() in self.snake.tolist()):
            self.done = True
            return self.get_observation(), -1.0, self.done, {}

        # 새 머리와 과일 사이의 거리 계산
        new_dist_to_fruit = np.linalg.norm(new_head - self.fruit)

        # 기본 보상 (생존 보상)
        reward = 0.0

        # 과일에 가까워지면 보상, 멀어지면 페널티 (단순화)
        if new_dist_to_fruit < old_dist_to_fruit:
            # 과일에 가까워지는 행동에 고정 보상
            reward += 0.1
        else:
            # 과일에서 멀어지는 행동에 고정 페널티
            reward -= 0.1

        # 과일을 먹으면 큰 보상
        if all(new_head == self.fruit):
            self.score += 1
            # 과일을 먹으면 큰 고정 보상 (단순화)
            reward = 1.0  # 고정 보상으로 단순화
            self.place_fruit()
        else:
            self.snake = self.snake[:-1, :]  # 꼬리 삭제

            # 벽 근처에서는 페널티 제거 (단순화)

            # 자기 몸과의 가장 가까운 거리 계산
            if len(self.snake) > 3:  # 몸 길이가 충분할 때만 계산
                min_body_dist = float('inf')
                for segment in self.snake[2:]:  # 머리 바로 다음 부분부터 계산
                    dist = np.linalg.norm(new_head - segment)
                    min_body_dist = min(min_body_dist, dist)

                # 몸에 너무 가까워지면 고정 페널티 (단순화)
                if min_body_dist < 1.5:  # 몸과의 거리가 1.5 이하이면 페널티
                    reward -= 0.1  # 고정 페널티

        # 새로운 머리 추가
        self.snake = np.concatenate([[new_head], self.snake], axis=0)
        return self.get_observation(), reward, self.done, {}

    def render(self, screen=None, mode='human'):
        import pygame
        if screen is not None:
            self.screen = screen
        elif not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode(
                (config.SCREEN_SIZE * config.PIXEL_SIZE, config.SCREEN_SIZE * config.PIXEL_SIZE)
            )
            pygame.display.set_caption('Snake Game with DQN')
        self.screen.fill((0, 0, 0))

        # 테두리 그리기
        pygame.draw.rect(self.screen, (255, 255, 255), [0, 0, config.SCREEN_SIZE * config.PIXEL_SIZE, config.LINE_WIDTH])
        pygame.draw.rect(self.screen, (255, 255, 255), [0, config.SCREEN_SIZE * config.PIXEL_SIZE - config.LINE_WIDTH, config.SCREEN_SIZE * config.PIXEL_SIZE, config.LINE_WIDTH])
        pygame.draw.rect(self.screen, (255, 255, 255), [0, 0, config.LINE_WIDTH, config.SCREEN_SIZE * config.PIXEL_SIZE])
        pygame.draw.rect(self.screen, (255, 255, 255), [config.SCREEN_SIZE * config.PIXEL_SIZE - config.LINE_WIDTH, 0, config.LINE_WIDTH, config.SCREEN_SIZE * config.PIXEL_SIZE])

        # 뱀과 과일 그리기
        for bit in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0),(
                bit[0] * config.PIXEL_SIZE,
                bit[1] * config.PIXEL_SIZE,
                config.PIXEL_SIZE,
                config.PIXEL_SIZE
            ))
        pygame.draw.rect(self.screen, (255, 0, 0), (
            self.fruit[0] * config.PIXEL_SIZE,
            self.fruit[1] * config.PIXEL_SIZE,
            config.PIXEL_SIZE,
            config.PIXEL_SIZE
        ))

        # 점수 표시
        font = pygame.font.SysFont(None, 20)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (5, 5))

        # 속도 정보 표시
        if self.speed_multiplier > 1.0:
            speed_text = font.render(f"Speed: {self.speed_multiplier}x", True, (255, 255, 0))
            self.screen.blit(speed_text, (5, 25))

        pygame.display.flip()

        # 속도 조절 (빠른 학습 모드에서는 디스플레이 업데이트 주기 조절)
        if self.speed_multiplier > 1.0:
            # 빠른 학습 모드에서는 디스플레이 업데이트 주기를 더 길게 설정
            pygame.time.wait(int(1000 / (config.FPS * self.speed_multiplier)))

    def close(self):
        if hasattr(self, 'screen'):
            import pygame
            pygame.quit()
            del self.screen
