import json
from collections import Counter
from matplotlib import pyplot as plt
import cv2 as cv2
import numpy as np

# Function that returns an array of groundtruth labels for an image. Used in determining true and false positives
def get_groundtruth_labels(filepath):
    with open(filepath, encoding='utf-8-sig') as json_file:
        data = json_file.read()
        data = json.loads(data)
        annotations = data['shapes']
        groundtruth_labels = []
        for item in annotations:
            object_class = item['label']
            groundtruth_labels.append(object_class)
        return groundtruth_labels


# Function to remove elements that occur less than k times
def removeElements(lst, k):
    counted = Counter(lst)

    temp_lst = []
    for el in counted:
        if counted[el] < k:
            temp_lst.append(el)

    res_lst = []
    for el in lst:
        # Want list with only one occurences of found objects that occur more than once
        if (el not in temp_lst) and (el not in res_lst):
            res_lst.append(el)
    return (res_lst)

def match_found_to_groundtruth(found_lst, groundtruth_lst):
    # Get true positives from & of two sets
    true_positives = set(found_lst) & set(groundtruth_lst)
    n_true_positives = len(true_positives)

    # Get false positives from items in found list but not in groundtruth list
    false_positives = set(found_lst) - set(groundtruth_lst)
    n_false_positives = len(false_positives)

    false_negatives = set(groundtruth_lst) - set(found_lst)
    n_false_negatives = len(false_negatives)

    return n_true_positives, n_false_positives, n_false_negatives, true_positives, false_positives, false_negatives

#def generate_bin_mask(image):


# Select bin patches to get patch for use w/ calc_histograms
def patch_selector(path_to_img):
    res_scale = 0.6  # rescale the input image if it's too large
    img = cv2.imread(path_to_img)
    img = cv2.resize(img, (0, 0), fx=res_scale, fy=res_scale)
    r = cv2.selectROI(img)
    imcrop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    cv2.destroyAllWindows()
    return imcrop


# Take in patch of image (type NDARRAY) and use it to get a plot of HSV values\
def calc_histograms(patch):
    img = patch
    color = ['b', 'g', 'r']
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    channels = ['Hue', 'Saturation', 'Value']
    plt.figure()
    for i, ch in enumerate(channels):
        hist = cv2.calcHist([hsv], [i], None, [256], [0, 256])
        plt.plot(hist, color=color[i])
    plt.title('HSV Histogram')
    plt.legend(channels)
    plt.show()


def show_convex_hull_bin(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerGreen = np.array([50, 0, 0])
    upperGreen = np.array([76, 255, 255])
    objmaskGreen = cv2.inRange(hsv, lowerGreen, upperGreen)
    kernel_5 = np.ones((5, 5), np.uint8)
    kernel_9 = np.ones((9,9), np.uint8)
    objmaskGreen = cv2.erode(objmaskGreen, kernel_9, iterations = 1)
    objmaskGreen = cv2.morphologyEx(objmaskGreen, cv2.MORPH_CLOSE, kernel=kernel_5)
    objmaskGreen = cv2.morphologyEx(objmaskGreen, cv2.MORPH_DILATE, kernel=kernel_5)
    im2, contours, hierarchy = cv2.findContours(objmaskGreen, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull = []
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False))
    # create an empty black image
    drawing = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (0, 255, 0)  # green - color for contours
        color = (255, 0, 0)  # blue - color for convex hull
        # draw ith contour
        cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 1, 8)
    cv2.imshow("Convex Hull in Blue, Contours in Green", drawing)
    cv2.waitKey(0)

def generate_bin_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowerGreen = np.array([50, 0, 0])
    upperGreen = np.array([76, 255, 255])
    objmaskGreen = cv2.inRange(hsv, lowerGreen, upperGreen)
    kernel_5 = np.ones((5, 5), np.uint8)
    kernel_9 = np.ones((9,9), np.uint8)
    kernel_21 = np.ones((21,21), np.uint8)
    objmaskGreen = cv2.erode(objmaskGreen, kernel_9, iterations = 1)
    objmaskGreen = cv2.morphologyEx(objmaskGreen, cv2.MORPH_CLOSE, kernel=kernel_5)
    objmaskGreen = cv2.morphologyEx(objmaskGreen, cv2.MORPH_DILATE, kernel=kernel_5)
    im2, contours, hierarchy = cv2.findContours(objmaskGreen, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull = []
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False))
    for i in range(len(hull)):
        cv2.fillPoly(objmaskGreen, pts = [hull[i]], color = (255,255,255))
    # Now Erode to Get Rid of Corners
    objmaskGreen = cv2.erode(objmaskGreen, kernel_21, iterations = 1)
    return objmaskGreen

