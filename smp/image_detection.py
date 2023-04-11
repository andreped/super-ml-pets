"""
Methods for performing detection of animals and items in the shop during deployment
"""

import pyautogui as gui
import cv2
import numpy as np
from PIL import ImageGrab, Image, ImageChops
import os
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from .utils import get_curr_screen_geometry


# global for all functions
paw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets/paw_icon.png")
arena_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets/arena_mode_icon.png")

paw_img = cv2.cvtColor(cv2.imread(paw_path, cv2.IMREAD_UNCHANGED)[..., :3], cv2.COLOR_BGR2RGB)
arena_img = cv2.cvtColor(cv2.imread(arena_path, cv2.IMREAD_UNCHANGED)[..., :3], cv2.COLOR_BGR2RGB)

# get screen resolution, store as global variable in this scope
curr_geometry = get_curr_screen_geometry()


def get_animal_from_screen():
    """
    captures images of the current animals on screen (to be classified at a later stage)
    """
    img_n_width = 130
    img = ImageGrab.grab(bbox=(450, 620, 1500, 750))  # bbox: left, top, right, bottom
    img_00 = img.crop((10, 0, 140, img_n_width))
    img_01 = img.crop((155, 0, 285, img_n_width))
    img_02 = img.crop((300, 0, 430, img_n_width))
    img_03 = img.crop((445, 0, 575, img_n_width))
    img_04 = img.crop((590, 0, 720, img_n_width))
    img_05 = img.crop((730, 0, 860, img_n_width))
    img_06 = img.crop((875, 0, 1005, img_n_width))
    images0 = [img_00, img_01, img_02, img_03, img_04, img_05, img_06]

    images = []
    for i in images0:
        images.append(cv2.cvtColor(np.array(i), cv2.COLOR_RGB2BGR))

    return images, images0


def matching(image, needle_img):
    """
    performs template matching to classify which animal/food/item it contains
    """
    result = cv2.matchTemplate(image, needle_img, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)

    if max_val > 0.7:
        return 1
        
    return 0


def get_image_directory(directory):
    """
    returns filenames one-by-one from SAP_res folder of animals and items
    """
    dir = os.listdir(directory)
    dir = list(filter((".DS_Store").__ne__, dir))  # remove occurences of ".DS_Store" relevant for macOS
    for folder in dir:
        for filename in os.listdir(os.path.join(directory, folder)):
            file = os.path.join(folder, filename)
            if os.path.isfile(os.path.join(directory, file)):
                yield os.path.join(directory, file).replace("\\", "/")


def find_the_animals(directory: str):
    """
    overall method for detecting which animals are on screen
    """
    list_of_animals = []
    images, references = get_animal_from_screen()

    # go through all the animals images in the directory
    for i in images:
        for j in get_image_directory(directory):
            im = cv2.imread(j, cv2.IMREAD_UNCHANGED)
            # matching returns which animals
            if matching(i, im):
                list_of_animals.append(j)
                break
    
    if len(list_of_animals) > 7:
        return 0

    list_of_animals1 = []
    for i in list_of_animals:
        temp = i.split('/')
        list_of_animals1.append(temp[-2])
        
    list_of_animals1 = tuple(list_of_animals1)
    references = tuple(references)

    if len(list_of_animals1) == 0:
        return list_of_animals1

    return list_of_animals1, references


def get_img_from_coords(coords):
    """
    method to get cropped image from coordinates
    """
    return np.array(ImageGrab.grab(bbox=coords))


def find_arena():
    """
    method to detect whether the game is in pre-arena state (game menu)
    """
    full_img = get_img_from_coords((310, 180, 1164, 671))
    value = ssim(arena_img, full_img, data_range=full_img.max() - full_img.min(), channel_axis=2)

    if value > 0.4:
        return True
    else:
        return False


def find_paw():
    """
    method to detect if the game is in pre-battle state (detects if paw icon is in top-right corner)
    """
    full_img = get_img_from_coords((1737.5, 15, 1812.5 + 4, 85 + 8))
    value = ssim(paw_img, full_img, data_range=full_img.max() - full_img.min(), channel_axis=2)

    if value > 0.4:
        return True
    else:
        return False
