import os
import pyautogui
import pytest
from smp.utils import define_logger

try:
    from pyvirtualdisplay.display import Display
    import Xlib

    # @TODO: This only works on Linux... Cannot run tests with virtual display
    disp = Display(visible=True, size=(1920, 1080), backend="xvfb", use_xauth=True)
    disp.start()
    pyautogui._pyautogui_x11._display = Xlib.display.Display(os.environ['DISPLAY'])
# Need to handle both regular and virtual displays for local and testing in the cloud 
except FileNotFoundError:  # pragma: no cover
    pass  # pragma: no cover

# need to define logger here, as some tests require logger to be defined,
# but it is only defined in main.py
define_logger()
