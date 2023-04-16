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
from .utils import get_screen_scale


# global for all functions
paw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets/paw_icon.png")
arena_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets/arena_mode_icon.png")

paw_img = cv2.cvtColor(cv2.imread(paw_path, cv2.IMREAD_UNCHANGED)[..., :3], cv2.COLOR_BGR2RGB)
arena_img = cv2.cvtColor(cv2.imread(arena_path, cv2.IMREAD_UNCHANGED)[..., :3], cv2.COLOR_BGR2RGB)

# get screen resolution scale, store as global variable in this scope
dimensions_scale = get_screen_scale()


def get_img_from_coords(coords, to_numpy=True):
    """
    method to get cropped image from coordinates
    """
    # as tuple does not support item assignment, change to np.array
    coords = np.array(coords)

    # scale coords
    coords[0] = coords[0] * dimensions_scale[0]
    coords[1] = coords[1] * dimensions_scale[1]
    coords[2] = coords[2] * dimensions_scale[0]
    coords[3] = coords[3] * dimensions_scale[1]

    coords = tuple(np.round(coords).astype(int))

    # snip image of screen from scaled coords
    img = ImageGrab.grab(bbox=coords)  # bbox = (left_x, top_y, right_x, bottom_y)

    if to_numpy:
        img = np.array(img)

    return img


def get_animal_from_screen():
    """
    captures images of the current animals on screen (to be classified at a later stage)
    """
    img = get_img_from_coords(coords=(450, 620, 1500, 750), to_numpy=False)  # bbox: left, top, right, bottom

    # template dimensions -> to be scaled if necessary
    img_n_width = 130
    bboxes = [
        [10, 0, 140, img_n_width],
        [155, 0, 285, img_n_width],
        [300, 0, 430, img_n_width],
        [445, 0, 575, img_n_width],
        [590, 0, 720, img_n_width],
        [730, 0, 860, img_n_width],
        [875, 0, 1005, img_n_width],
    ]

    images = []
    images0 = []
    for bbox in bboxes:
        bbox[0] = dimensions_scale[0] * bbox[0]
        bbox[1] = dimensions_scale[1] * bbox[1]
        bbox[2] = dimensions_scale[0] * bbox[2]
        bbox[3] = dimensions_scale[1] * bbox[3]
        
        image = img.crop(tuple(bbox))
        images0.append(image)
        images.append(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

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


def find_arena():
    """
    method to detect whether the game is in pre-arena state (game menu)
    """
    full_img = get_img_from_coords((310, 180, 1164, 671))
    try:
        value = ssim(arena_img, full_img, data_range=full_img.max() - full_img.min(), channel_axis=2)
    except RuntimeWarning:
        # this is not a critical error. We can disregard this and continue running
        value = 0

    if value > 0.4:
        return True
    else:
        return False


def find_paw():
    """
    method to detect if the game is in pre-battle state (detects if paw icon is in top-right corner)
    """
    full_img = get_img_from_coords((1737.5, 15, 1816.5, 93))
    print(full_img.shape)
    print(paw_img.shape)
    #if full_img.shape != paw_img.shape:

    try:
        value = ssim(paw_img, full_img, data_range=full_img.max() - full_img.min(), channel_axis=2)
    except RuntimeWarning:
        # this is not a critical error. We can disregard this and continue running
        value = 0

    if value > 0.4:
        return True
    else:
        return False
