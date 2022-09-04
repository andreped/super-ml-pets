from src.train_agent import opponent_generator
from src.utils import define_logger


def test_gen():
    opponent_generator(25)


def test_logger():
    define_logger(1)



