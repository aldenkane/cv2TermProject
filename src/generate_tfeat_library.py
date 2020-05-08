# generate_tfeat_library.py
# Notre Dame CSE 40536/60536
# Generate library of tfeat descriptors for detection in later images
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

#######################################
# Section 1: Declare Globals and File Structure, Init TFEAT Model w/ Weights
#######################################

# Directory Structure Info
OBJECTS_DIR_PATH = '../objects/T3'
OBJECTS_FOLDERS = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10', 'o11']
CLASS_NAMES = ['Paintbrush', 'Spray_Sunscreen', 'Rub_Sunscreen', 'Dice_Container', 'Tape', 'Cetaphil', 'Sunglasses', 'Pillbottle', 'Fuzzy', 'Marker', 'Frisbee']
DESIRED_SAMPLE = 'mobi'

#######################################
# Section 2: Resize Image Parameters, SIFT Library Parameters, Initiate tfeat, BRISK, FLANN Matcher
#######################################

# Resize Scale, Images Sampled
RES_SCALE_OBJECT = 0.2
N_IMAGES_SAMPLED = 1

# Initiate tfeat and BRISK
tfeat = tfeat_model.TNet()
models_path = '../pretrained-models'
net_name = 'tfeat-liberty'
mag_factor = 3
tfeat.load_state_dict(torch.load(os.path.join(models_path,net_name+".params")))
tfeat.cuda()
tfeat.eval()
brisk = cv2.BRISK_create()

# Globals for Iterating Through Objects Folders
CURRENT_ITEM = 0

#######################################
# Section 3: Randomly Sample Images, Iterate Through Directory
#######################################
for folder in OBJECTS_FOLDERS:
    # Get to Desired Folder
    current_path = OBJECTS_DIR_PATH + os.path.sep + folder + os.path.sep + DESIRED_SAMPLE
    # Maintain list of sampled photos that resets for each class
    sampled_photos = []
    # Create a name for numpy descriptor array
    descriptor_filepath_to_save = '../tfeat_descriptor_library/' + str(N_IMAGES_SAMPLED) + os.path.sep + str(N_IMAGES_SAMPLED) + '_' + CLASS_NAMES[CURRENT_ITEM] + '_TFEAT_DESC.npy'
    # Create Array to Save
    array_to_save = np.empty((0,128), dtype=np.float32)

    # Get three random images from current path
    for samples in range(N_IMAGES_SAMPLED):
        picture = random.choice(os.listdir(current_path))
        # Ensure it's a .jpg and not already been sampled
        while not ((picture.endswith('.jpg')) and (picture not in sampled_photos)):
            picture = random.choice(os.listdir(current_path))
        # Append the sampled photo to the library of sampled photos to ensure that photos are distinct
        sampled_photos.append(picture)

        # Should probably make sure pictures are distinct, or hand curate photos. But, I'm lazy and time constraints apply here
        # If file is a .jpg, read in OPENCV grayscale and get keypoints
        pic_path = current_path + os.path.sep + picture
        img = cv2.imread(str(pic_path), cv2.IMREAD_GRAYSCALE)

        # Rescale
        img = cv2.resize(img, (0,0), fx = RES_SCALE_OBJECT, fy = RES_SCALE_OBJECT)
        key_pts, descriptors = brisk.detectAndCompute(img, None)
        desc_tfeat = tfeat_utils.describe_opencv(tfeat, img, key_pts, 32, mag_factor)

        # Append Array to Itself for Saving
        array_to_save = np.vstack((array_to_save, desc_tfeat))

    # Save Array after Getting Desired Number of Images
    print(CLASS_NAMES[CURRENT_ITEM])
    np.save(descriptor_filepath_to_save, array_to_save)
    # Iterate on class
    CURRENT_ITEM = CURRENT_ITEM + 1

