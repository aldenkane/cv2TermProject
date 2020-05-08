# CSE 40536/60536 Term Project: Object Detection in a Dense Cluster

## Report

### Methods and Models

**Alden's Keypoint-Based Approach**

I was motivated to attempt solutions that did not require deep learning for object detection.  I also wanted to incorporate traditional computer vision methods to achieve higher detection accuracy. Methods and models that I used are:
1. Segmenting for the tote based on color
2. SIFT keypoint descriptor matching
3. CNN-based keypoint descriptor matching

As this project presented object detection in a constrained environment (i.e. bin of known size and color), using this information to occlude background noise (e.g. floor around bin, carpet, edges of the bin) is a practical means of removing spurious detections. I achieved this by segmenting the image for all 'H' values in the HSV color space within a range of [50,76]. The function `generate_bin_mask(img)` in `alden_cv2_functions.py`. It takes an RGB image as an argument, and returns a 1-channel mask for the bin.

![Figure 1. Generated Masks for Totes](/report_images/masks.png)

 
## Instructions for Running Programs

## Consent for Amazon Robotics

## Division of Work