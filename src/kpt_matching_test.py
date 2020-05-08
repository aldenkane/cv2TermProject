# kpt_matching_test.py
# Notre Dame CSE 40536/60536
# Performs SIFT/TFEAT Matching of Objects in a Bin on a Test Image
# Must use OpenCV 3.4.2, as SIFT/SURF support has been deprecated in later versions
# Team of Alden Kane and Xing Jie Zhong
#   Author: Alden Kane

import cv2 as cv2
import numpy as np
import os
import random
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
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import time
from alden_cv2_functions import get_groundtruth_labels, removeElements, match_found_to_groundtruth, generate_bin_mask

#######################################
# Section 1: Declare Globals
#######################################

TRAIN_OBJECT_SAMPLES = 9
CLASS_NAMES = ['Paintbrush', 'Spray_Sunscreen', 'Rub_Sunscreen', 'Dice_Container', 'Tape', 'Cetaphil', 'Sunglasses', 'Pillbottle', 'Fuzzy', 'Marker', 'Frisbee']
SIFT_DESCRIPTOR_FILES = []
TFEAT_DESCRIPTOR_FILES = []
RES_SCALE_BIN = 0.6
PATH_TO_TEST_JPGS = '../collection/mobi_test_jpgs'
PATH_TO_TEST_JSON = '../collection/mobi_test_json'
LOWE_THRESHOLD = 0.7
IGNORE_THRESHOLD = 2
BIN_IMG_PATH = '../collection/mobi_test_jpgs/l2_al1_m_side_4.jpg'

# Assemble List of SIFT Descriptor Files
for i, obj in enumerate(CLASS_NAMES):
    file_to_load = '../sift_descriptor_library/' + str(TRAIN_OBJECT_SAMPLES) + os.path.sep + str(TRAIN_OBJECT_SAMPLES) + '_' + str(CLASS_NAMES[i]) + '_' + 'SIFT_DESC.npy'
    SIFT_DESCRIPTOR_FILES.append(file_to_load)

# Assemble List of tfeat Descriptor Files
for i, obj in enumerate(CLASS_NAMES):
    file_to_load = '../tfeat_descriptor_library/' + str(TRAIN_OBJECT_SAMPLES) + os.path.sep + str(TRAIN_OBJECT_SAMPLES) + '_' + str(CLASS_NAMES[i]) + '_' + 'TFEAT_DESC.npy'
    TFEAT_DESCRIPTOR_FILES.append(file_to_load)

#######################################
# Section 2: Initiate SIFT, tfeat, FLANN
#######################################

# Initiate SIFT
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.06, edgeThreshold = 10)

# Initiate tfeat
tfeat = tfeat_model.TNet()
models_path = '../pretrained-models'
net_name = 'tfeat-liberty'
mag_factor = 3
tfeat.load_state_dict(torch.load(os.path.join(models_path,net_name+".params")))
tfeat.cuda()
tfeat.eval()

# Initiate FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

#######################################
# Section 3: Read Image, Get SIFT Keypoints
#######################################

image_color = cv2.imread(str(BIN_IMG_PATH))
image_color = cv2.resize(image_color, (0, 0), fx=RES_SCALE_BIN, fy=RES_SCALE_BIN)
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
bin_mask = generate_bin_mask(image_color)
# Get bin kpts and descriptors using sift
bin_kpts, bin_desc = sift.detectAndCompute(image, mask=bin_mask)

# Create List to Store Found Items In
sift_found_items = []

#######################################
# Section 4: Iterate through SIFT Descriptor Libraries and Perform Matching
##33#####################################
for i, lib in enumerate(SIFT_DESCRIPTOR_FILES):
    obj_desc = np.load(str(SIFT_DESCRIPTOR_FILES[i]))
    matches = flann.knnMatch(obj_desc, bin_desc, k=2)
    # store all the good matches as per Lowe's ratio test.
    for m, n in matches:
        if m.distance < LOWE_THRESHOLD * n.distance:
            sift_found_items.append(CLASS_NAMES[i])

# Once all descriptors have been matched and had Lowe's Ratio Test Applied
conf_sift_found_items = removeElements(sift_found_items, k=2)
print('SIFT Keypoint Detection Found: ' + str(conf_sift_found_items))

#######################################
# Section 5: Iterate Through Tfeat Descriptor Libraries and Perform Matching
##33#####################################

# Describe bin key pts using tfeat
bin_desc_tfeat = tfeat_utils.describe_opencv(tfeat, image, bin_kpts, 32, mag_factor)
# List for Found Items
tfeat_found_items = []

for i, lib in enumerate(TFEAT_DESCRIPTOR_FILES):
    obj_desc = np.load(str(TFEAT_DESCRIPTOR_FILES[i]))
    matches = flann.knnMatch(obj_desc, bin_desc_tfeat, k=2)
    # store all the good matches as per Lowe's ratio test.
    for m, n in matches:
        if m.distance < LOWE_THRESHOLD * n.distance:
            tfeat_found_items.append(CLASS_NAMES[i])

# Once all descriptors have been matched and had Lowe's Ratio Test Applied, print detections
conf_tfeat_found_items = removeElements(tfeat_found_items, k=2)
print('CNN-Based Keypoint Detection Found: ' + str(conf_tfeat_found_items))

# Show Images at End
plt.imshow(bin_mask,), plt.show()
plt.imshow(image_color,), plt.show()
