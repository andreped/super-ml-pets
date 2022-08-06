<p align="center">
  <img src="https://www.gamelivestory.com/images/article/super-auto-pets-how-to-level-up-quickly-main.webp" width="50%" alt='super-auto-pets'>
</p>

# Super-Auto-Pets

[![License](https://img.shields.io/badge/License-GPLv3-lightgray.svg)](https://opensource.org/licenses/GPLv3)
![CI](https://github.com/andreped/super-ml-pets/workflows/test/badge.svg)

Using Reinforcement Learning (RL) to train AIs for Super Auto Pets. Can train AIs in simulated environment or directly in the real game (the latter is quite bad atm, but works).

Tested and supports Python 3.6-3.10 and works cross-platform (Ubuntu, Windows, macOS).

### Setup

1. Clone the repo:
```
git clone https://github.com/andreped/super-ml-pets.git
```

2. Create virtual environment:
```
cd super-ml-pets/
virtualenv -ppython3 venv --clear
source venv/bin/activate
```

3. Install requirements:
```
pip install -r requirements.txt
```

4. Need to install Tesseract separately (depends on operating system). See [here](https://github.com/UB-Mannheim/tesseract/wiki).

### Usage
For training in simulated environment, run:
```bash
cd V6.0/
python DQN.py
```

### Troubleshoot
If you are working on Windows, you need to use slightly different commands for setting up the environment. If you do not have virtualenv in the path, you need to do:
```
python -m virtualenv -ppython3 venv --clear
```

To activate virtual env on windows do:
```
./venv/Scripts/activate
```

### Acknowledgements

This implementation is based on the work by [HJK-Z](https://github.com/HJK-Z/Super-Auto-Pets), with aim to improve the design with smarter reinforcement learning schemes.
The game logic is built around [sapai](https://github.com/manny405/sapai).
