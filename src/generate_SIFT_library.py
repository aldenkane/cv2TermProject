# sift_Playground.py
# Notre Dame CSE 40536/60536
# Generate library of SIFT keypoints for detection in later images
# Must use OpenCV 3.4.2, as SIFT/SURF support has been deprecated in later versions
# Team of Alden Kane and Xing Jie Zhong
#   Author: Alden Kane

# NOTES for Code:
# -Important to sample object images from similar size as they would appear in the bin images, so the keypoints are a similar scale
# -Playing with scale of keypoint detector is crucial for making it better
# -Need items in directory to account for rotation and such
# -A mask that ignores the white background will give me stronger keypoints and ignore keypoints from the white
# -Want to get rid of spurious matches so use Lowe's Ratio Test

import cv2 as cv2
import numpy as np
import os
import random
from matplotlib import pyplot as plt

#######################################
# Section 1: Declare Globals and File Structure
#######################################

# Directory Structure Info
OBJECTS_DIR_PATH = '../objects/T3'
OBJECTS_FOLDERS = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'o9', 'o10', 'o11']
CLASS_NAMES = ['Paintbrush', 'Spray_Sunscreen', 'Rub_Sunscreen', 'Dice_Container', 'Tape', 'Cetaphil', 'Sunglasses', 'Pillbottle', 'Fuzzy', 'Marker', 'Frisbee']
DESIRED_SAMPLE = 'mobi'

#######################################
# Section 2: Resize Image Parameters, SIFT Library Parameters, Initiate SIFT
#######################################

# Resize Scale, Images Sampled
RES_SCALE_OBJECT = 0.2
N_IMAGES_SAMPLED = 9

# Initiate Sift w/ modified thresholds to ignore high frequency noise
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.06, edgeThreshold = 10)

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
    descriptor_filepath_to_save = '../sift_descriptor_library/' + str(N_IMAGES_SAMPLED) + os.path.sep + str(N_IMAGES_SAMPLED) + '_' + CLASS_NAMES[CURRENT_ITEM] + '_SIFT_DESC.npy'
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
        key_pts, descriptors = sift.detectAndCompute(img, None)

        # Append Array to Itself for Saving
        array_to_save = np.vstack((array_to_save, descriptors))

        # Optional: Display Images
        # img = cv2.drawKeypoints(img, key_pts, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # plt.imshow(img,), plt.show()

    # Save Array after Getting Desired Number of Images
    print(CLASS_NAMES[CURRENT_ITEM])
    np.save(descriptor_filepath_to_save, array_to_save)
    # Iterate on class
    CURRENT_ITEM = CURRENT_ITEM + 1

