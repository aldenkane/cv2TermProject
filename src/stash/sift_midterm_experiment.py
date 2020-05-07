# sift_Playground.py
# Notre Dame CSE 40536/60536
# Playing w/ sift codes before implementing them in experiments for object detection in a dense cluster
# Must use OpenCV 3.4.2, as SIFT/SURF support has been deprecated in later versions
# Team of Alden Kane and Xing Jie Zhong
#   Author: Alden Kane

import numpy as np
import cv2 as cv2
import os
import csv

#######################################
# Section 1: Declare Globals, File Structure
#######################################

# Declare Folders
objects_folder = '/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/objects/T3/mft_objects'
bins_folder = '/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/collection/mft_bins'
path_to_img_1 = '/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/objects/T3/o1/mobi/m_Paintbrush1.jpg'
path_to_img_2 = '/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/objects/T3/o2/mobi/m_Spray_Sunscreen1.jpg'
# Directory with all photos
directory = r'/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/objects/T3/mft_objects'

# Declare Resize Scale
res_scale = 0.3

# Create logs folder
log_directory = r'../sift_logs'

# Initialize Global
classified = 0

# Initiate Sift and flann Matcher
sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

current_image = 0
len_directory = len(os.listdir(directory))

master_csv_log = str(log_directory) + '/master_csv_log.csv'

#######################################
# Section 1.5 For Loop Structure
#######################################
# First for loop to compare to others
for filename_1 in os.listdir(directory):
    if filename_1.endswith(".jpg"):
        current_image = current_image + 1
        # Set path to first image
        path_to_img_1 = os.path.join(directory, filename_1)

        # Read first image, resize, and convert to grayscale
        img_1 = cv2.imread(str(path_to_img_1))
        img_1 = cv2.resize(img_1, (0, 0), fx=res_scale, fy=res_scale)
        gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

        # Run SIFT for 1st Image
        key_pts_1, descriptors_1 = sift.detectAndCompute(gray_1, None)

        # Make a Log File for this Test
        log_name_txt = str(log_directory) + '/log_txt_' + str(filename_1[:-4]) + '.txt'
        log_name_csv = str(log_directory) + '/log_csv_' + str(filename_1[:-4]) + '.csv'

        print(str(current_image) + 'of' + str(len_directory))

        for filename_2 in os.listdir(directory):
            # Read second image, resize, and convert to grayscale
            path_to_img_2 = os.path.join(directory, filename_2)
            img_2 = cv2.imread(str(path_to_img_2))
            img_2 = cv2.resize(img_2, (0, 0), fx=res_scale, fy=res_scale)
            gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

            # Run SIFT Second Image
            key_pts_2, descriptors_2 = sift.detectAndCompute(gray_2, None)

            # Compare the Two
            matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
            # store all the good matches as per Lowe's ratio test.
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append('Matched ' + str(filename_1) + ' to ' + str(filename_2) + ' with ' + str(m))
                    # See if objects are the same
                    if filename_1[2:6] == filename_2[2:6]:
                        classified = 'True Positive'
                    else:
                        classified = 'False Positive'

            # Write to a log csv file
            # Write to a log csv file
            with open(str(log_name_csv), 'a') as csvfile:
                fieldnames = ['True/False', '# of Matched Descriptors']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                # Only write if there were keypoints matched, for lack of erroneous false positives
                num_positives = len(good)
                if len(good) != 0:
                    writer.writerow({'True/False': str(classified), '# of Matched Descriptors': str(num_positives)})

            # Write to a log txt file
            with open(str(log_name_txt), 'a') as txtfile:
                if len(good) != 0:
                    txtfile.write(str(classified) + ' --> Matched: (' + str(filename_1) + ', ' + str(filename_2) + ')' + '\n')

            # Write to a master log csv file
            with open(str(master_csv_log), 'a') as master_csvfile:
                fieldnames = ['True/False', '# of Matched Descriptors']
                master_writer = csv.DictWriter(master_csvfile, fieldnames=fieldnames)
                # Only write if there were keypoints matched, for lack of erroneous false positives
                num_positives = len(good)
                if len(good) != 0:
                    master_writer.writerow({'True/False': str(classified), '# of Matched Descriptors': str(num_positives)})
