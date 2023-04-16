import pyautogui
import os
from pyvirtualdisplay.display import Display
import Xlib

# @TODO: This only works on Linux... Cannot run tests with virtual display
disp = Display(visible=True, size=(1920, 1080), backend="xvfb", use_xauth=True)
disp.start()
pyautogui._pyautogui_x11._display = Xlib.display.Display(os.environ['DISPLAY'])

from smp.utils import define_logger, opponent_generator, custom_easeOutQuad, \
    move_drag_tween, get_position, get_curr_screen_geometry, get_screen_scale
import numpy as np


def test_position():
    get_position()


def test_get_geometry():
    get_curr_screen_geometry()


def test_get_scale():
    get_screen_scale()


def test_gen():
    opponent_generator(25)


def test_logger():
    for i in range(5):
        try:
            define_logger(i)
        except ValueError as e:
            if i != 4:
                raise e


def test_easeOutQuad():
    [custom_easeOutQuad(x) for x in np.linspace(0, 1, 1000)]


def test_move_drag_tween():
    [move_drag_tween(x) for x in np.linspace(0, 1, 1000)]
