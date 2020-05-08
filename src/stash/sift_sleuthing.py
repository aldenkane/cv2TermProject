# sift_Playground.py
# Notre Dame CSE 40536/60536
# Playing w/ sift codes before implementing them in experiments for object detection in a dense cluster
# Must use OpenCV 3.4.2, as SIFT/SURF support has been deprecated in later versions
# Team of Alden Kane and Xing Jie Zhong
#   Author: Alden Kane

import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt

#######################################
# Section 1: Declare Globals
#######################################

objects_folder = '/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/objects/T3/mft_objects'
bins_folder = '/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/collection/mft_bins'
res_scale_bin = 0.5
res_scale_object = 0.2

# Dice Container vs.
path_to_img_1 = '/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/collection/l1/al1/mobi/side/3.jpg'
path_to_img_2 = '/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/objects/T3/mft_objects/m_Dice_Container9.jpg'

# Frisbee, Sunscreen, Pillbottle, and Tape are all Shown Well

# Really good sunscreen, cetaphil in here
path_to_img_1 = '/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/collection/l1/pl1/mobi/top/3.jpg'
# Frisbee for Matching Here
path_to_img_2 = '/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/objects/T3/o11/mobi/m_Frisbee3.jpg'
# Sunscreen w/ back, upside down
path_to_img_2 = '/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/objects/T3/o2/mobi/m_Spray_Sunscreen3.jpg'

#######################################
# Section 2: Read Images and Preprocess
#######################################

# Read first image, resize, and convert to grayscale
img_1 = cv2.imread(str(path_to_img_1))
img_1 = cv2.resize(img_1, (0,0), fx = res_scale_bin, fy = res_scale_bin)
gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

# Read second image, resize, and convert to grayscale
img_2 = cv2.imread(str(path_to_img_2))
img_2 = cv2.resize(img_2, (0,0), fx = res_scale_object, fy = res_scale_object)
gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

# Initiate Sift
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold = 0.06, edgeThreshold = 10)

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

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

# Write to a log file
with open('sift_log.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % m for m in good)

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img_1, key_pts_1, img_2, key_pts_2, matches, None, **draw_params)

plt.imshow(img3,), plt.show()

# Write Images for Debug
cv2.imwrite('keypoints1.jpg', img_1)
cv2.imwrite('keypoints2.jpg', img_2)

### Notes for Report

# Having a 'bin' image twice the size of an 'object' image introduces keypoints at similar scales
# Glares can create a match