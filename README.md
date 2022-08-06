<p align="center">
  <img src="https://www.gamelivestory.com/images/article/super-auto-pets-how-to-level-up-quickly-main.webp" width="50%" alt='super-auto-pets'>
</p>

# Super-Auto-Pets
Developing AIs for Super Auto Pets

[![License](https://img.shields.io/badge/License-GPLv3-lightgray.svg)](https://opensource.org/licenses/GPLv3)
![CI](https://github.com/andreped/super-ml-pets/workflows/test/badge.svg)

**Current status**: Using Deep Q-Learning to train AI. Simulated training works well, real world training quite bad...

V1.0: Trains AI directly with the game (plays against real people)

V6.0: Smarter training scheme, training against itself.

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

### Troubleshoot
If you are working on Windows, you need to use slightly different commands for setting up the environment. If you do not have virtualenv in the path, you need to do:
```
python -m virtualenv -ppython3 venv --clear
```

To activate virtual env on windows do:
```
./venv/Scripts/activate
```

### To Do:

1. Use V6.0 to train AI against itself using Q-Learning
2. Use V1.0 to test trained AI in real game
3. Git gud
4. ...
5. $$

### Acknowledgements

This implementation is based on the work by [HJK-Z](https://github.com/HJK-Z/Super-Auto-Pets), with aim to improve the design with smarter reinforcement learning schemes.
The game logic is built around [sapai](https://github.com/manny405/sapai).
