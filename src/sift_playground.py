# sift_Playground.py
# Notre Dame CSE 40536/60536
# Playing w/ sift codes before implementing them in experiments for object detection in a dense cluster
# Must use OpenCV 3.4.2, as SIFT/SURF support has been deprecated in later versions
# Team of Alden Kane and Xing Jie Zhong
#   Author: Alden Kane

import numpy as np
import cv2 as cv2

#######################################
# Section 1: Declare Globals
#######################################

objects_folder = '/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/objects/T3/mft_objects'
bins_folder = '/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/collection/mft_bins'
path_to_img_1 = '/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/objects/T3/mft_objects/m_Dice_Container6.jpg'
path_to_img_2 = '/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/objects/T3/mft_objects/c_Fuzzy10.jpg'
res_scale = 0.3

#######################################
# Section 2: Read Images and Preprocess
#######################################

# Read first image, resize, and convert to grayscale
img_1 = cv2.imread(str(path_to_img_1))
img_1 = cv2.resize(img_1, (0,0), fx = res_scale, fy = res_scale)
gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

# Read second image, resize, and convert to grayscale
img_2 = cv2.imread(str(path_to_img_2))
img_2 = cv2.resize(img_2, (0,0), fx = res_scale, fy = res_scale)
gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

# Initiate Sift
sift = cv2.xfeatures2d.SIFT_create()

#######################################
# Section 2: Initiate SIFT
#######################################

# Find Keypoints and Descriptors
key_pts_1, descriptors_1 = sift.detectAndCompute(gray_1, None)
key_pts_2, descriptors_2 = sift.detectAndCompute(gray_2, None)

# Draw Keypoints
img_1 = cv2.drawKeypoints(gray_1, key_pts_1, img_1)
img_2 = cv2.drawKeypoints(gray_2, key_pts_2, img_2)

#######################################
# Section 3: FLANN Based Matching
#######################################

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

# store all the good matches as per Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

# Write to a log file
with open('sift_log.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % m for m in good)

# Write Images for Debug
cv2.imwrite('keypoints1.jpg', img_1)
cv2.imwrite('keypoints2.jpg', img_2)