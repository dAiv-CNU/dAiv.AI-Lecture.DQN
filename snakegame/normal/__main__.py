import pygame

from .snake import SnakeBoard, SCREEN_SIZE, PIXEL_SIZE


def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_SIZE * PIXEL_SIZE, SCREEN_SIZE * PIXEL_SIZE))
    pygame.display.set_caption("Snake Game")

    game = SnakeBoard(screen)
    while game.run():
        pass

    pygame.quit()


if __name__ == '__main__':
    main()
