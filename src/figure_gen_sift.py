# l2b_sift_experiment.py
# Takes library of SIFT descriptors by loading numpy arrays and matches them to bin images
# Notre Dame Computer Vision 2 Term Project
#   Author: Alden Kane

import cv2 as cv2
import os
import random
from matplotlib import pyplot as plt
import numpy as np
import sys
from alden_cv2_functions import get_groundtruth_labels, removeElements, match_found_to_groundtruth, generate_bin_mask

#######################################
# Section 1: Declare Globals
#######################################

DESIRED_OBJECT_SAMPLES = 1
CLASS_NAMES = ['Paintbrush', 'Spray_Sunscreen', 'Rub_Sunscreen', 'Dice_Container', 'Tape', 'Cetaphil', 'Sunglasses', 'Pillbottle', 'Fuzzy', 'Marker', 'Frisbee']
OBJECTS_FOLDERS = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10', 'o11']
RES_SCALE_BIN = 0.45
RES_SCALE_OBJ = 0.2
PATH_TO_TEST_JPGS = '../collection/mobi_test_jpgs'
PATH_TO_TEST_JSON = '../collection/mobi_test_json'
OBJECTS_DIR_PATH = '../objects/T3'
LOWE_THRESHOLD = 0.6
IGNORE_THRESHOLD = 2
DESIRED_SAMPLE = 'mobi'


# See if o1, o2, o3
DESIRED_OBJECT_CLASS = 'o9'
OBJECT_PATH = OBJECTS_DIR_PATH + os.path.sep + DESIRED_OBJECT_CLASS

#######################################
# Section 2: Initiate SIFT and FLANN Matchers
#######################################

sift = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.04, edgeThreshold = 10)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

#######################################
# Section 3: Select a Random Bin, Random Objects, Detect and Compute SIFT, Match Keypoints
#######################################

# Bin Images
bin_choice = random.choice(os.listdir(PATH_TO_TEST_JPGS))
bin_path = PATH_TO_TEST_JPGS + os.path.sep + bin_choice
bin_image_color = cv2.imread(str(bin_path))
bin_image_color = cv2.resize(bin_image_color, (0,0), fx = RES_SCALE_BIN, fy = RES_SCALE_BIN)
bin_image = cv2.cvtColor(bin_image_color, cv2.COLOR_BGR2GRAY)
bin_mask = generate_bin_mask(bin_image_color)

# Get Image to Sample
sampled_photos = []
object_path = str(OBJECTS_DIR_PATH) + os.path.sep + str(DESIRED_OBJECT_CLASS) + os.path.sep + str(DESIRED_SAMPLE)

for samples in range(DESIRED_OBJECT_SAMPLES):
    picture = random.choice(os.listdir(object_path))
    # Ensure it's a .jpg and not already been sampled
    while not ((picture.endswith('.jpg')) and (picture not in sampled_photos)):
        picture = random.choice(os.listdir(object_path))
    # Append the sampled photo to the library of sampled photos to ensure that photos are distinct
    picture_path = OBJECTS_DIR_PATH + os.path.sep + DESIRED_OBJECT_CLASS + os.path.sep + DESIRED_SAMPLE + os.path.sep + picture
    sampled_photos.append(picture_path)

object_1_image = cv2.imread(str(sampled_photos[0]), cv2.IMREAD_GRAYSCALE)
object_1_image = cv2.resize(object_1_image, (0,0), fx = RES_SCALE_OBJ, fy = RES_SCALE_OBJ)

bin_kpts, bin_desc = sift.detectAndCompute(bin_image, mask = bin_mask)
object_1_kpts, object_1_desc = sift.detectAndCompute(object_1_image, None)

# Use flann matcher
matches = flann.knnMatch(object_1_desc, bin_desc, k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < LOWE_THRESHOLD*n.distance:
        good.append(m)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < LOWE_THRESHOLD*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

# Swapped bin and Obj
img3 = cv2.drawMatchesKnn(object_1_image, object_1_kpts, bin_image, bin_kpts, matches, None, **draw_params)

plt.imshow(bin_mask,), plt.show()
plt.imshow(img3,), plt.show()