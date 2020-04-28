import cv2
import pickle

im = cv2.imread("/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/collection/l1/pl2/c615/side/2.jpg")
index = pickle.loads(open("/Users/aldenkane1/Documents/1College/4SenSem2/Computer Vision 2/cv2TermProject/src/kp_db.txt", "rb").read())
kp = []

for point in index:
    temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
    kp.append(temp)

# Draw the keypoints
imm = cv2.drawKeypoints(im, kp, im)
cv2.imshow("Image", imm)
cv2.waitKey(0)
