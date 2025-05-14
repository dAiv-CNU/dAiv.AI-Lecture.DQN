import pygame
import random
import numpy as np
import time
from config import FPS, SCREEN_SIZE, PIXEL_SIZE, LINE_WIDTH, SPEED

# 방향 정의
DIRECTIONS = np.array([
    (0, -1),  # UP
    (1, 0),   # RIGHT
    (0, 1),   # DOWN
    (-1, 0)   # LEFT
])

class Snake:
    def __init__(self, screen):
        self.screen = screen
        self.speed = SPEED
        self.reset_game()

    def reset_game(self):
        center_x, center_y = SCREEN_SIZE // 2, SCREEN_SIZE // 2
        self.snake = np.array([
            [center_x, center_y],
            [center_x, center_y + 1],
            [center_x, center_y + 2],
            [center_x, center_y + 3]
        ])
        self.direction = 0
        self.score = 0
        self.last_move_time = time.time()
        self.place_fruit()

    def place_fruit(self):
        while True:
            x = random.randint(0, SCREEN_SIZE-1)
            y = random.randint(0, SCREEN_SIZE-1)
            if list([x, y]) not in self.snake.tolist():
                break
        self.fruit = np.array([x, y])

    def move(self):
        old_head = self.snake[0]
        movement = DIRECTIONS[self.direction]
        new_head = old_head + movement

        if (new_head[0] < 0 or
                new_head[0] >= SCREEN_SIZE or
                new_head[1] < 0 or
                new_head[1] >= SCREEN_SIZE or
                new_head.tolist() in self.snake.tolist()):
            return False

        if all(new_head == self.fruit):
            self.score += 1
            self.place_fruit()
        else:
            self.snake = self.snake[:-1, :]

        self.snake = np.concatenate([[new_head], self.snake], axis=0)
        return True

    def run(self):
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 20)
        font.set_bold(True)

        apple_image = pygame.Surface((PIXEL_SIZE, PIXEL_SIZE))
        apple_image.fill((0, 255, 0))
        snake_image = pygame.Surface((PIXEL_SIZE, PIXEL_SIZE))
        snake_image.fill((255, 0, 0))

        game_over = False

        while not game_over:
            clock.tick(FPS)
            current_time = time.time()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    elif event.key == pygame.K_SPACE:
                        pause = True
                        while pause:
                            for e in pygame.event.get():
                                if e.type == pygame.QUIT:
                                    return False
                                elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                                    pause = False

            # 방향키 입력 처리
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] and self.direction != 2:
                self.direction = 0
            elif keys[pygame.K_RIGHT] and self.direction != 3:
                self.direction = 1
            elif keys[pygame.K_DOWN] and self.direction != 0:
                self.direction = 2
            elif keys[pygame.K_LEFT] and self.direction != 1:
                self.direction = 3

            # 뱀 이동 (속도에 따라)
            if current_time - self.last_move_time >= 1.0/self.speed:
                if not self.move():
                    game_over = True
                    break
                self.last_move_time = current_time

            # 화면 그리기
            self.screen.fill((0, 0, 0))

            # 테두리 그리기
            pygame.draw.rect(self.screen, (255,255,255), [0,0,SCREEN_SIZE*PIXEL_SIZE,LINE_WIDTH])
            pygame.draw.rect(self.screen, (255,255,255), [0,SCREEN_SIZE*PIXEL_SIZE-LINE_WIDTH,SCREEN_SIZE*PIXEL_SIZE,LINE_WIDTH])
            pygame.draw.rect(self.screen, (255,255,255), [0,0,LINE_WIDTH,SCREEN_SIZE*PIXEL_SIZE])
            pygame.draw.rect(self.screen, (255,255,255), [SCREEN_SIZE*PIXEL_SIZE-LINE_WIDTH,0,LINE_WIDTH,SCREEN_SIZE*PIXEL_SIZE+LINE_WIDTH])

            # 뱀과 과일 그리기
            for bit in self.snake:
                self.screen.blit(snake_image, (bit[0] * PIXEL_SIZE, bit[1] * PIXEL_SIZE))
            self.screen.blit(apple_image, (self.fruit[0] * PIXEL_SIZE, self.fruit[1] * PIXEL_SIZE))

            # 점수와 속도 표시
            score_text = font.render(f"Score: {self.score}", False, (255, 255, 255))
            speed_text = font.render(f"Speed: {self.speed}", False, (255, 255, 255))
            self.screen.blit(score_text, (5, 5))
            self.screen.blit(speed_text, (5, 25))

            pygame.display.update()

        # 게임 오버 화면
        game_over_font = pygame.font.SysFont(None, 48)
        game_over_text = game_over_font.render("Game Over!", True, (255, 0, 0))
        score_text = game_over_font.render(f"Final Score: {self.score}", True, (255, 255, 255))
        restart_text = font.render("Press SPACE to restart or ESC to quit", True, (255, 255, 255))

        text_rect = game_over_text.get_rect(center=(SCREEN_SIZE*PIXEL_SIZE/2, SCREEN_SIZE*PIXEL_SIZE/2 - 50))
        score_rect = score_text.get_rect(center=(SCREEN_SIZE*PIXEL_SIZE/2, SCREEN_SIZE*PIXEL_SIZE/2))
        restart_rect = restart_text.get_rect(center=(SCREEN_SIZE*PIXEL_SIZE/2, SCREEN_SIZE*PIXEL_SIZE/2 + 50))

        self.screen.blit(game_over_text, text_rect)
        self.screen.blit(score_text, score_rect)
        self.screen.blit(restart_text, restart_rect)
        pygame.display.update()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    elif event.key == pygame.K_SPACE:
                        self.reset_game()
                        return True

def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_SIZE * PIXEL_SIZE, SCREEN_SIZE * PIXEL_SIZE))
    pygame.display.set_caption('Snake Game')

    game = Snake(screen)
    while game.run():
        pass

    pygame.quit()

if __name__ == '__main__':
    main()