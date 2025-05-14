# dAiv Reinforcement Learning Basics

## Dependencies
To run this project, ensure Python 3.10 or later is installed. Follow the steps below to set up the environment:


## Reproduction
### Clone this repo
```bash
git clone https://github.com/dAiv-CNU/dAiv.AI-Lecture.DQN dqn
cd dqn
```

### Install Required Packages
```bash
pip install --upgrade uv
uv sync
```

### Install PyTorch
For NVIDIA GPUs (CUDA versions):

CUDA 11.8
```bash
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --frozen
```

CUDA 12.1
```bash
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --frozen
```

CUDA 12.4
```bash
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --frozen
```

using Mac or Window cpu environment
```bash
uv add torch torchvision torchaudio
```

### How to Run
#### 1. Play the Snake Game
Run the playable snake game:
```bash
python snake.py normal
```

#### 2. Evolutionary Snake Game
Use genetic algorithms to evolve the snake:
```bash
python snake.py genetic
```

#### 3. Deep Q-Learning Snake Game
Run the DQN-based snake AI:
```bash
python snake.py dqn
```


## Credits
Snake game code by HonzaKral: https://gist.github.com/HonzaKral/833ee2b30231c53ec78e
