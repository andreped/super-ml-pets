"""
Methods for performing detection of animals and items in the shop during deployment
"""

import pyautogui as gui
import cv2
import numpy as np
from PIL import ImageGrab, Image, ImageChops
import os
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from .utils import get_screen_scale
import tensorflow as tf


# global for all functions
paw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets/paw_icon.png")
arena_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets/arena_mode_icon.png")

paw_img = cv2.cvtColor(cv2.imread(paw_path, cv2.IMREAD_UNCHANGED)[..., :3], cv2.COLOR_BGR2RGB)
arena_img = cv2.cvtColor(cv2.imread(arena_path, cv2.IMREAD_UNCHANGED)[..., :3], cv2.COLOR_BGR2RGB)

# Load the pre-trained CNN model
model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(112, 112, 3))
# model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(112, 112, 3))

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

    needle_img = cv2.resize(needle_img, image.shape[:2][::-1])

    needle_img = needle_img[..., ::-1]
    image = image[..., ::-1]

    #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #needle_img = cv2.cvtColor(needle_img, cv2.COLOR_RGB2GRAY)

    #needle_img = (needle_img > 220).astype("uint8")
    #image = (image > 220).astype("uint8")

    #hd95 = hausdorff_distance(image, needle_img, method="modified")
    #hd95 = dice(image, needle_img)

    #max_val = hd95

    #mask = np.zeros_like(needle_img.copy())#[..., 0]
    #mask[needle_img > 0] = 1
    #mask = needle_img.copy()
    # mask = 1 - mask

    # pad image to fit needle image into
    #tmp = np.zeros((int(image.shape[0] * 3), int(image.shape[1] * 3), 3), dtype="uint8")
    #tmp[image.shape[0]:(2*image.shape[0]), image.shape[1]:(2*image.shape[1]), :] = image
    #image = tmp.copy()

    """

    # Initialize the ORB detector and detect the keypoints in the query image and scene
    orb = cv2.ORB_create()
    query_keypoints, query_descriptors = orb.detectAndCompute(image, None)
    scene_keypoints, scene_descriptors = orb.detectAndCompute(needle_img, None)

    # Match the keypoints using Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(query_descriptors, scene_descriptors)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # if these match, there should be a lot of matches,
    # use that as criteria to state whether it is a match

    # Calculate the distance between the matched keypoints
    distance = np.mean([match.distance for match in matches])
    """

    # Preprocess the images
    image_pr = cv2.resize(image, (112, 112))
    #image_pr = Image.resize(image_pr, (112, 112))
    #image_pr = imutils.resize(image, width=112)
    image_pr = tf.keras.applications.vgg16.preprocess_input(image_pr)

    needle_img_pr = cv2.resize(needle_img, (112, 112))
    #needle_img = Image.resize(needle_img, (112, 112))
    #needle_img_pr = imutils.resize(needle_img, width=112)
    needle_img_pr = tf.keras.applications.vgg16.preprocess_input(needle_img_pr)

    # Extract the features from the images
    query_features = model.predict(np.expand_dims(image_pr, axis=0), verbose=False)
    scene_features = model.predict(np.expand_dims(needle_img_pr, axis=0), verbose=False)

    # Calculate the distance between the features -> this worked OK but not thaaat well
    #distance = np.linalg.norm(query_features - scene_features)

    # Calculate the cosine similarity between the features
    similarity = cosine_similarity(query_features.reshape(1, -1), scene_features.reshape(1, -1))
    distance = similarity[0][0]

    # Print the similarity score
    print(f"The similarity score between the two images is {similarity[0][0]}")


    print(distance)

    # needle_img[needle_img == 0] = 255
    #result = cv2.matchTemplate(image, needle_img, cv2.TM_SQDIFF_NORMED) # cv2.TM_SQDIFF_NORMED)  # cv2.TM_CCOEFF_NORMED)  # TM_CCORR_NORMED
    #min_val, max_val, _, _ = cv2.minMaxLoc(result)
    # print(max_val)

    #print(min_val, max_val)

    if distance > 0.52: # distance < 1300:
        #"""
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(image)
        ax[1].imshow(needle_img)
        #ax[1].set_title(str(min_val) + " | " + str(max_val))
        ax[1].set_title(str(distance))
        #ax[2].imshow(mask)
        plt.show()
        #"""

        return 1

    #if max_val > 0.7:
    #    return 1

    if False: # (distance < 57.5) and (len(matches) > 8):
        #for m in matches:
        #    print(m.distance)

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(image)
        ax[1].imshow(needle_img)
        #ax[1].set_title(str(min_val) + " | " + str(max_val))
        ax[1].set_title(str(distance))
        #ax[2].imshow(mask)
        plt.show()
        return 1

    #if min_val < 0.75:
    #    return 1
        
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
    pet_paths = [directory + filename for filename in os.listdir(directory)]
    print("N IMAGES:", len(images))
    for image in images:
        #plt.imshow(image)
        #plt.show()
        for filename in os.listdir(directory):
            pet_path = directory + filename
            im = cv2.imread(pet_path, cv2.COLOR_BGR2RGB)[..., :3][:, ::-1, :]
            # im = cv2.resize(im, (150, 150))
            
            # matching returns which animals
            if matching(image, im):
                list_of_animals.append("pet-" + filename.split(".")[0])
                break
    
    if len(list_of_animals) > 7:
        return 0

    #list_of_animals1 = []
    #for i in list_of_animals:
    #    temp = i.split('/')
    #    list_of_animals1.append(temp[-2])
    list_of_animals1 = list_of_animals.copy()

    print(list_of_animals)
        
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

    if full_img.shape != arena_img.shape:
        full_img = cv2.resize(full_img, arena_img.shape[:2][::-1])

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
    
    if full_img.shape != paw_img.shape:
        full_img = cv2.resize(full_img, paw_img.shape[:2][::-1])

    try:
        value = ssim(paw_img, full_img, data_range=full_img.max() - full_img.min(), channel_axis=2)
    except RuntimeWarning:
        # this is not a critical error. We can disregard this and continue running
        value = 0

    if value > 0.4:
        return True
    else:
        return False
