from src.utils import define_logger, opponent_generator


def test_gen():
    opponent_generator(25)


def test_logger():
    for i in range(4):
        define_logger(i)
