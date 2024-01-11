"""
Convenience methods
"""

import tkinter as tk
import pynput
import numpy as np
import logging as log
import os
from sapai_gym.opponent_gen.opponent_generators import biggest_numbers_horizontal_opp_generator


def define_logger(verbose=1):
    """
    method which sets the verbose handler
    """
    if verbose == 0:
        level = None
    elif verbose == 1:
        level = log.INFO
    elif verbose == 2:
        level = log.DEBUG
    elif verbose == 3:
        level = log.WARNING
    else:
        raise ValueError("Unknown verbose was set. 0 to disable verbose, 1 for INFO, 2 for DEBUG, 3 for WARNING.")

    log.basicConfig(
        format="%(levelname)s %(filename)s %(lineno)s %(message)s",
        level=level
        )


def get_position():
    """
    returns locations of relevant objects in the game
    """
    template_resolution = [1920, 1080]  # height, width

    # hard-coded positions assuming a screen resolution of 1920 x 1080
    position = {
        '0_team_slot': (524, 408),
        '1_team_slot': (669, 408),
        '2_team_slot': (810, 408),
        '3_team_slot': (950, 408),
        '4_team_slot': (1090, 408),
        '0_slot': (529, 695),
        '1_slot': (667, 695),
        '2_slot': (812, 695),
        '3_slot': (952, 695),
        '4_slot': (1092, 695),
        '5_slot': (1232, 695),
        '6_slot': (1372, 695),
        'freeze': (1029, 981),
        'end_turn': (1614, 978),
        'roll': (193, 978),
        'sell': (1029, 981),
        # other stuff related to image_detection.py (detection of animals)
        'img_lefttop': (450, 620),
        'img_bottomright': (1500, 750),
        'img_00': (10, 140),
        'img_01': (155, 285),
        'img_02': (300, 285),
        'img_03': (445, 575),
        'img_04': (590, 720),
        'img_05': (730, 860),
        'img_06': (875, 1005),
    }

    # scale these positions to match current screen resolution 
    # disable scaling for now
    curr_geometry = get_curr_screen_geometry()
    log.info("Current display dimensions: (" + str(curr_geometry[0]) + ", " + str(curr_geometry[1]) + ")")
    for key in position.keys():
        curr_position = position[key]
        # # note that there is an intended height/width switch!
        position[key] = (int(np.round(curr_position[0] * float(curr_geometry[0]) / template_resolution[0])),
                         int(np.round(curr_position[1] * float(curr_geometry[1]) / template_resolution[1])))

    return position


def get_screen_scale():
    """
    Returns dimension scale difference between current screen dimensions and template.
    Defined as scale = curr_screen_dims / template_dims
    """
    template_resolution = [1920, 1080]  # height, width
    curr_geometry = get_curr_screen_geometry()

    return np.array(curr_geometry) / np.array(template_resolution)


def get_curr_screen_geometry():
    """
    Workaround to get the size of the current screen in a multi-screen setup
    Note that this method captures the current scaled resolution and not necessary the true resolution
    """
    root = tk.Tk()
    root.update_idletasks()
    root.attributes('-fullscreen', True)
    root.state('iconic')
    geometry = root.winfo_geometry()
    root.destroy()
    return np.array(geometry.split("+")[0].split("x")).astype(int)


def move_drag_tween(n):
    if n == 0:
        return 0.0
    else:
        return 1 + 0.05 * np.log10(n)


def custom_easeOutQuad(n):
    """A quadratic tween function that begins fast and then decelerates.
    Args:
      n (float): The time progress, starting at 0.0 and ending at 1.0.
    Returns:
      (float) The line progress, starting at 0.0 and ending at 1.0. Suitable for passing to getPointOnLine().
    """
    return -n * (n-2)


def opponent_generator(num_turns):
    """
    returns teams to fight against in the gym - value set to 25
    """
    return biggest_numbers_horizontal_opp_generator(25)


def kill_process(key):
    """
    method to stop agent from running if 'escape' key is pressed
    """
    global stop_program
    if key == pynput.keyboard.Key.esc:
        print("\nEscape pressed, stopping agent...")
        stop_program = True
        return False
