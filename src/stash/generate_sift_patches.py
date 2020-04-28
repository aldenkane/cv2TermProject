#!/usr/bin/env python
from sklearn.feature_extraction import image
import numpy as np
import cv2
import os

# Folder for the generated patches.
patch_folder = '../patches'

# Image domain (and folder where source images are located).
db_folder = '../objects/T3/proto_set'
image_domain = 'objects'
classes_folder = db_folder + os.path.sep + image_domain

# Desired scales of the filters to be generated; it must relate to the available source image patches.
patch_scales = 64

# Seed to make patch creation deterministic
random_state = 3011

# Number of patches per image
max_patches_per_image = 30

for image_file in os.listdir(db_folder):

    # Image Loading
    im_path = db_folder + os.path.sep + image_file
    one_image = cv2.imread(im_path)
    one_image_gray = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)

    # Initialize Sift Detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Run Sift to Get Keypoints
    key_pts, descriptors = sift.detectAndCompute(one_image_gray, None)

    # Question for Czajka --> How to Know which keypoints to select?
    for pts in key_pts:
        x, y = pts.pt[0], pts.pt[1]

    patches = image.extract_patches_2d(
        image = one_image,
        patch_size = (scale, scale),
        max_patches = max_patches_per_image,
        random_state = random_state
    )

    # creates the output scale folder, if necessary
    scale_path = patch_folder + os.path.sep + image_domain + os.path.sep + '{:02d}'.format(scale)
    if not os.path.exists(scale_path):
        os.makedirs(scale_path)

    for idx, patch in enumerate(patches):
        patch_patch = scale_path + os.path.sep + image_file[:-4] + '_{:06d}'.format(idx) + '.png'
        cv2.imwrite(patch_patch, patch)

    print("Generated for images in: {}".format(db_folder))

print('*** DONE! ***')