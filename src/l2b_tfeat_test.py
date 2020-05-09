# l2b_sift_experiment.py
# Takes library of SIFT descriptors by loading numpy arrays and matches them to bin images
# Notre Dame Computer Vision 2 Term Project
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

# Get Time
start_time = time.time()

TRAIN_OBJECT_SAMPLES = 9
CLASS_NAMES = ['Paintbrush', 'Spray_Sunscreen', 'Rub_Sunscreen', 'Dice_Container', 'Tape', 'Cetaphil', 'Sunglasses', 'Pillbottle', 'Fuzzy', 'Marker', 'Frisbee']
DESCRIPTOR_FILES = []
RES_SCALE_BIN = 0.6
PATH_TO_TEST_JPGS = '../test_sample'
PATH_TO_TEST_JSON = '../collection/mobi_test_json'
LOWE_THRESHOLD = 0.6
IGNORE_THRESHOLD = 2

# Assemble List of Descriptor Files
for i, obj in enumerate(CLASS_NAMES):
    file_to_load = '../tfeat_descriptor_library/' + str(TRAIN_OBJECT_SAMPLES) + os.path.sep + str(TRAIN_OBJECT_SAMPLES) + '_' + str(CLASS_NAMES[i]) + '_' + 'TFEAT_DESC.npy'
    DESCRIPTOR_FILES.append(file_to_load)

# Globals for Experimental Analysis
POSITIVES = 0

#######################################
# Section 2: Initiate tfeat and FLANN Matchers
#######################################

# Initiate tfeat and BRISK
tfeat = tfeat_model.TNet()
models_path = '../pretrained-models'
net_name = 'tfeat-liberty'
mag_factor = 3
tfeat.load_state_dict(torch.load(os.path.join(models_path,net_name+".params")))
tfeat.cuda()
tfeat.eval()
brisk = cv2.BRISK_create()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

#######################################
# Section 3: Iterate through Test Database
#######################################

# Iterate Through Test Database
for item in os.listdir(PATH_TO_TEST_JPGS):
    if item.endswith('.JPG'):
        # Set path to image, read it w/ opencv
        path_to_im = os.path.join(PATH_TO_TEST_JPGS, item)
        image_color = cv2.imread(str(path_to_im))
        image_color = cv2.resize(image_color, (0,0), fx = RES_SCALE_BIN, fy = RES_SCALE_BIN)
        image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        bin_mask = generate_bin_mask(image_color)
        # Get bin kpts and descriptors using sift
        bin_kpts, bin_desc = brisk.detectAndCompute(image, mask = None)
        bin_desc_tfeat = tfeat_utils.describe_opencv(tfeat, image, bin_kpts, 32, mag_factor)
        # Create List to Store Found Items In
        found_items = []

        #######################################
        # Section 4: Iterate through SIFT Descriptor Libraries and Perform Matching
        ##33#####################################
        for i,lib in enumerate(DESCRIPTOR_FILES):
            obj_desc = np.load(str(DESCRIPTOR_FILES[i]))
            print('Loaded Desc File')
            matches = flann.knnMatch(obj_desc, bin_desc_tfeat, k=2)
            # store all the good matches as per Lowe's ratio test.
            for m, n in matches:
                if m.distance < LOWE_THRESHOLD * n.distance:
                    found_items.append(CLASS_NAMES[i])

        # Once all descriptors have been matched and had Lowe's Ratio Test Applied, get TP and FP
        conf_found_items = removeElements(found_items, k=2)
        num_positives = len(conf_found_items)
        POSITIVES = POSITIVES + num_positives

#######################################
# Section 5: Print Precision, Recall, Save to Log
#######################################

with open('../sift_logs/test_set_masked.txt', 'a') as file:
    sys.stdout = file
    print('OBJ --> BIN, Tfeat')
    print('Training Object Samples: ' + str(TRAIN_OBJECT_SAMPLES))
    print('Ratio Test Value: ' + str(LOWE_THRESHOLD))
    print('Ignored Matches w/ Less than K Occurrences, K: ' + str(IGNORE_THRESHOLD))
    print('Positives: ' + str(POSITIVES))
    print('------------------------------------------')

print('Finished Program, Positives:' + str(POSITIVES))

