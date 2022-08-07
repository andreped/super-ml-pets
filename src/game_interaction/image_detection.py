"""
Script for performing detection of animals and items in the shop
"""
import pyautogui as gui
import cv2 as cv
import numpy as np
from PIL import ImageGrab, Image, ImageChops
import os

def get_animal_from_screen():
    img = ImageGrab.grab(bbox=(450, 620, 1500, 750))
    img_00 = img.crop((10,0,140,130))
    img_01 = img.crop((155,0,285,130))
    img_02 = img.crop((300,0,430,130))
    img_03 = img.crop((445,0,575,130))
    img_04 = img.crop((590,0,720,130))
    img_05 = img.crop((730,0,860,130))
    img_06 = img.crop((875,0,1005,130))
    images0 = [img_00, img_01, img_02, img_03, img_04, img_05,img_06]
    images = []
    for i in images0:
        images.append(cv.cvtColor(np.array(i), cv.COLOR_RGB2BGR))
    return images, images0

def matching(image, needle_img):
    result = cv.matchTemplate(image,needle_img, cv.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv.minMaxLoc(result)
    # print(max_val)
    if max_val > 0.7:
        return 1
    return 0

# returns the filename one by one
def get_image_directory(directory):
    # print(directory)
    dir = os.listdir(directory)
    for folder in dir:
        for filename in os.listdir(os.path.join(directory, folder)):
            file = os.path.join(folder, filename)
            if os.path.isfile(os.path.join(directory,file)):
                yield os.path.join(directory,file)

def find_the_animals(directory = '.\\SAP_res\\'):
    list_of_animals = []
    images, references = get_animal_from_screen()
    #go through all the animals images in the directory
    for i in images:
        for j in get_image_directory(directory):
            im = cv.imread(j, cv.IMREAD_UNCHANGED)
            #matching returns which animals
            if matching(i, im):
                list_of_animals.append(j)
                break
    if len(list_of_animals) > 7:
        return 0
    list_of_animals1 = []
    for i in list_of_animals:
        temp = i.split('\\')
        list_of_animals1.append(temp[2])
    list_of_animals1 = tuple(list_of_animals1)
    print(list_of_animals1)
    references = tuple(references)
    if len(list_of_animals1) == 0:
        return list_of_animals1
    return list_of_animals1, references
