from smp.utils import define_logger, opponent_generator, custom_easeOutQuad, move_drag_tween
import numpy as np


def test_gen():
    opponent_generator(25)


def test_logger():
    for i in range(4):
        define_logger(i)


def test_easeOutQuad():
    [custom_easeOutQuad(x) for x in np.linspace(0, 1, 1000)]


def test_move_drag_tween():
    [move_drag_tween(x) for x in np.linspace(0, 1, 1000)]
