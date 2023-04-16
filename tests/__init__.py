import pyautogui
import os

try:
    from pyvirtualdisplay.display import Display
    import Xlib

    # @TODO: This only works on Linux... Cannot run tests with virtual display
    disp = Display(visible=True, size=(1920, 1080), backend="xvfb", use_xauth=True)
    disp.start()
    pyautogui._pyautogui_x11._display = Xlib.display.Display(os.environ['DISPLAY'])
except FileNotFoundError:  # Need to handle both regular and virtual displays for local and testing in the cloud
    pass

# @TODO: Need to be able to gracefully kill the virtual display when finished
