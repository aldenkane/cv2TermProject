import cv2
import pickle

im = cv2.imread("/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/collection/l1/pl2/c615/side/2.jpg")
gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# Initiate Sift w/ modified thresholds to ignore high frequency noise
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.06, edgeThreshold=10)
kp, desc = sift.detectAndCompute(gr, None)

index = []
for point in kp:
    temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
    index.append(temp)

# Dump the keypoints
f = open("/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/src/kp_db.txt", "ab")
f.write(pickle.dumps(index))
f.close()
