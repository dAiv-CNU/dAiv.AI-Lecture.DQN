import pygame
import os, random
import numpy as np
from config import SCREEN_SIZE, PIXEL_SIZE, FPS, LINE_WIDTH


DIRECTIONS = np.array([
  (0, -1), # UP
  (1, 0), # RIGHT
  (0, 1), # DOWN
  (-1, 0) # LEFT
])

class Snake():
  snake, fruit = None, None

  def __init__(self, s, genome):
    self.genome = genome

    self.s = s
    self.score = 0
    # self.snake = np.array([[15, 26], [15, 27], [15, 28], [15, 29]])
    center_x, center_y = SCREEN_SIZE // 2, SCREEN_SIZE // 2
    self.snake = np.array([[center_x, center_y], [center_x, center_y + 1], [center_x, center_y + 2], [center_x, center_y + 3]])
    self.direction = random.randint(0, 3)  # 무작위 방향으로 초기화 (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
    self.place_fruit()
    self.timer = 0
    self.last_fruit_time = 0

    # fitness
    self.fitness = 0.
    self.last_dist = np.inf

  def place_fruit(self, coord=None):
    if coord:
      self.fruit = np.array(coord)
      return

    while True:
      x = random.randint(0, SCREEN_SIZE-1)
      y = random.randint(0, SCREEN_SIZE-1)
      if list([x, y]) not in self.snake.tolist():
        break
    self.fruit = np.array([x, y])

  def step(self, direction):
    old_head = self.snake[0]
    movement = DIRECTIONS[direction]
    new_head = old_head + movement

    if (
        new_head[0] < 0 or
        new_head[0] >= SCREEN_SIZE or
        new_head[1] < 0 or
        new_head[1] >= SCREEN_SIZE or
        new_head.tolist() in self.snake.tolist()
      ):
      # self.fitness -= config.FPS/2
      return False

    # eat fruit
    if all(new_head == self.fruit):
      self.last_fruit_time = self.timer
      self.score += 1
      self.fitness += 10
      self.place_fruit()
    else:
      tail = self.snake[-1]
      self.snake = self.snake[:-1, :]

    self.snake = np.concatenate([[new_head], self.snake], axis=0)
    return True

  def get_inputs(self):
    head = self.snake[0]
    result = [1., 1., 1., 0., 0., 0.]

    # check forward, left, right
    possible_dirs = [
      DIRECTIONS[self.direction], # straight forward
      DIRECTIONS[(self.direction + 3) % 4], # left
      DIRECTIONS[(self.direction + 1) % 4] # right
    ]

    # 0 - 1 ... danger - safe
    for i, p_dir in enumerate(possible_dirs):
      # sensor range = 5
      for j in range(5):
        guess_head = head + p_dir * (j + 1)

        if (
          guess_head[0] < 0 or
          guess_head[0] >= SCREEN_SIZE or
          guess_head[1] < 0 or
          guess_head[1] >= SCREEN_SIZE or
          guess_head.tolist() in self.snake.tolist()
        ):
          result[i] = j * 0.2
          break

    # finding fruit
    # 과일 방향 감지 로직 개선
    fruit_dir = self.fruit - head

    # 직진 방향에 과일이 있는지 확인
    forward_dot = np.dot(fruit_dir, possible_dirs[0])
    if forward_dot > 0:  # 과일이 앞쪽에 있음
      result[3] = 1

    # 왼쪽 방향에 과일이 있는지 확인
    left_dot = np.dot(fruit_dir, possible_dirs[1])
    if left_dot > 0:  # 과일이 왼쪽에 있음
      result[4] = 1

    # 오른쪽 방향에 과일이 있는지 확인
    right_dot = np.dot(fruit_dir, possible_dirs[2])
    if right_dot > 0:  # 과일이 오른쪽에 있음
      result[5] = 1

    return np.array(result)

  def run(self):
    self.fitness = 0

    prev_key = pygame.K_UP

    font = pygame.font.SysFont(None, 20)
    font.set_bold(True)
    appleimage = pygame.Surface((PIXEL_SIZE, PIXEL_SIZE))
    appleimage.fill((0, 255, 0))
    img = pygame.Surface((PIXEL_SIZE, PIXEL_SIZE))
    img.fill((255, 0, 0))
    clock = pygame.time.Clock()

    while True:
      self.timer += 0.1
      if self.fitness < -FPS/2 or self.timer - self.last_fruit_time > 0.1 * FPS * 5:
        # self.fitness -= FPS/2
        print('Terminate!')
        break

      clock.tick(FPS)
      for e in pygame.event.get():
        if e.type == pygame.QUIT:
          pygame.quit()
        elif e.type == pygame.KEYDOWN:
          # QUIT
          if e.key == pygame.K_ESCAPE:
            pygame.quit()
            exit()
          # PAUSE
          if e.key == pygame.K_SPACE:
            pause = True
            while pause:
              for ee in pygame.event.get():
                if ee.type == pygame.QUIT:
                  pygame.quit()
                elif ee.type == pygame.KEYDOWN:
                  if ee.key == pygame.K_SPACE:
                    pause = False
          if __name__ == '__main__':
            # CONTROLLER
            if prev_key != pygame.K_DOWN and e.key == pygame.K_UP:
              self.direction = 0
              prev_key = e.key
            elif prev_key != pygame.K_LEFT and e.key == pygame.K_RIGHT:
              self.direction = 1
              prev_key = e.key
            elif prev_key != pygame.K_UP and e.key == pygame.K_DOWN:
              self.direction = 2
              prev_key = e.key
            elif prev_key != pygame.K_RIGHT and e.key == pygame.K_LEFT:
              self.direction = 3
              prev_key = e.key

      # action
      if __name__ != '__main__':
        inputs = self.get_inputs()
        outputs = self.genome.forward(inputs)
        outputs = np.argmax(outputs)

        if outputs == 0: # straight
          pass
        elif outputs == 1: # left
          self.direction = (self.direction + 3) % 4
        elif outputs == 2: # right
          self.direction = (self.direction + 1) % 4

      if not self.step(self.direction):
        break

      # compute fitness
      current_dist = np.linalg.norm(self.snake[0] - self.fruit)
      if self.last_dist > current_dist:
        self.fitness += 1.
      else:
        self.fitness -= 1.5
      self.last_dist = current_dist

      self.s.fill((0, 0, 0))
      pygame.draw.rect(self.s, (255,255,255), [0,0,SCREEN_SIZE*PIXEL_SIZE,LINE_WIDTH])
      pygame.draw.rect(self.s, (255,255,255), [0,SCREEN_SIZE*PIXEL_SIZE-LINE_WIDTH,SCREEN_SIZE*PIXEL_SIZE,LINE_WIDTH])
      pygame.draw.rect(self.s, (255,255,255), [0,0,LINE_WIDTH,SCREEN_SIZE*PIXEL_SIZE])
      pygame.draw.rect(self.s, (255,255,255), [SCREEN_SIZE*PIXEL_SIZE-LINE_WIDTH,0,LINE_WIDTH,SCREEN_SIZE*PIXEL_SIZE+LINE_WIDTH])
      for bit in self.snake:
        self.s.blit(img, (bit[0] * PIXEL_SIZE, bit[1] * PIXEL_SIZE))
      self.s.blit(appleimage, (self.fruit[0] * PIXEL_SIZE, self.fruit[1] * PIXEL_SIZE))
      score_ts = font.render(str(self.score), False, (255, 255, 255))
      self.s.blit(score_ts, (5, 5))
      pygame.display.update()

    return self.fitness, self.score
