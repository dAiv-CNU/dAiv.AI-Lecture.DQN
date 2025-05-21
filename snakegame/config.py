"""
snakegame/config.py
Game Configuration File
"""

#
# Game Overall Config
FPS = 60
SCREEN_SIZE = 40
PIXEL_SIZE = 20
LINE_WIDTH = 1
SPEED = 25  # 초당 이동 횟수 (초기 속도)

#
# Genetic Parameters
N_POPULATION = 50
N_BEST = 5
N_CHILDREN = 5
PROB_MUTATION = 0.4

#
# DQN Hyper-Parameters
BATCH_SIZE = 128       # 리플레이 메모리 배치 크기
GAMMA = 0.95           # 할인율
EPS_START = 1.0        # 엡실론 시작 값
EPS_END = 0.05         # 엡실론 최종 값
EPS_DECAY = 2000       # 엡실론 감소 속도
TAU = 0.005            # 타겟 네트워크 업데이트 속도
LR = 1e-4              # 학습률
WEIGHT_DECAY = 1e-5    # 가중치 감소
NUM_EPISODES = 100000  # 총 학습 에피소드 수
