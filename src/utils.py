"""
Convenience methods
"""

import tkinter as tk
import numpy as np


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
    ''' # disable scaling for now
    curr_geomentry = get_curr_screen_geometry()
    for key in position.keys():
        curr_position = position[key]
        # # note that there is an intended height/width switch!
        position[key] = (int(np.round(curr_position[0] * float(curr_geomentry[1]) / template_resolution[0])),
                         int(np.round(curr_position[1] * float(curr_geomentry[0]) / template_resolution[1])))
    '''
    return position


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
