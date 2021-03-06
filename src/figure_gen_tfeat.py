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
import torchvision as tv
import phototour
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import math
import tfeat_model
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import tfeat_utils
import random
from matplotlib import pyplot as plt
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
LOWE_THRESHOLD = 0.8
IGNORE_THRESHOLD = 2
DESIRED_SAMPLE = 'mobi'

# See if o1, o2, o3
DESIRED_OBJECT_CLASS = 'o5'
OBJECT_PATH = OBJECTS_DIR_PATH + os.path.sep + DESIRED_OBJECT_CLASS

# Get Random Number for End of Image
append = random.random()

#######################################
# Section 2: Initiate tfeat, BRISK, and FLANN Matchers
#######################################

# Init Tfeat
tfeat = tfeat_model.TNet()
models_path = '../pretrained-models'
net_name = 'tfeat-liberty'
tfeat.load_state_dict(torch.load(os.path.join(models_path,net_name+".params")))
tfeat.cuda()
tfeat.eval()

# Initiate BRISK Matcher
brisk = cv2.BRISK_create()

#######################################
# Section 3: Select a Random Bin, Random Objects, Detect and Compute SIFT, Match Keypoints
#######################################

# Bin Images
bin_choice = random.choice(os.listdir(PATH_TO_TEST_JPGS))
bin_path = PATH_TO_TEST_JPGS + os.path.sep + bin_choice
bin_image_color = cv2.imread(str(bin_path))
#bin_image_color = cv2.resize(bin_image_color, (0,0), fx = RES_SCALE_BIN, fy = RES_SCALE_BIN)
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
#object_1_image = cv2.resize(object_1_image, (0,0), fx = RES_SCALE_OBJ, fy = RES_SCALE_OBJ)

bin_kpts, bin_desc = brisk.detectAndCompute(bin_image, mask = bin_mask)
object_1_kpts, object_1_desc = brisk.detectAndCompute(object_1_image, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(object_1_desc, bin_desc, k=2)
# Apply ratio test
good = []
for m, n in matches:
    if m.distance < LOWE_THRESHOLD * n.distance:
        good.append([m])

img_brisk = cv2.drawMatchesKnn(object_1_image, object_1_kpts, bin_image, bin_kpts, good, 0, flags=2)
im_brisk_filename = '../tfeat_logs/images/tfeat_figure_gen_BRISK_' + str(append) + '.jpg'
cv2.imwrite(str(im_brisk_filename), img_brisk)

# Get tfeat Descriptors
mag_factor = 3
bin_tfeat_desc = tfeat_utils.describe_opencv(tfeat, bin_image, bin_kpts, 32, mag_factor)
obj_tfeat_desc = tfeat_utils.describe_opencv(tfeat, object_1_image, object_1_kpts, 32, mag_factor)

print('tfeat Bin Descriptors')
print(bin_tfeat_desc)
print('-----------------------')
print('tfeat Object Descriptors')
print(obj_tfeat_desc)

# Update Matcher
bf = cv2.BFMatcher(cv2.NORM_L2)

# Match tfeat descriptors
matches = bf.knnMatch(obj_tfeat_desc, bin_tfeat_desc, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < LOWE_THRESHOLD * n.distance:
        good.append([m])

tfeat_img = cv2.drawMatchesKnn(object_1_image, object_1_kpts, bin_image, bin_kpts, good, 0, flags=2)

im_name = '../tfeat_logs/images/tfeat_figure_gen_cnn_' + str(append) + '.jpg'
cv2.imwrite(str(im_name), tfeat_img)
