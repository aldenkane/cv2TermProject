# Creates directories for CV2 data collection
# part 2 (object capture) and starts data capture from camera
# Usage:
# python objects.py
# python objects.py --dirs-only

import os
import sys
import math
import time
import itertools

# PARAMETERS
BASE_PATH = 'objects'
VIDEO_DEVICE = 1        # switch cameras here (0, 1, ...)


def create_dirs():

    dirs = {
        'teams': [
            'T1',
            'T2',
            'T3',
            'T4',
            'T5',
            'T6',
            'T7',
        ],
        'objects': [
            'o1', 'o2', 'o3', 'o4', 'o5',
            'o6', 'o7', 'o8', 'o9', 'o10',
            'o11', 'o12', 'o13', 'o14', 'o15',
        ],
        'cams': [
            'c615',
            'mobi',
        ],
    }

    paths = list(itertools.product(dirs['teams'], dirs['objects'], dirs['cams']))
    paths = [ os.path.join(*p) for p in paths ]

    for p in paths:

        path = os.path.join(BASE_PATH, p)
        print("Creating {}".format(path))
        os.makedirs(path, exist_ok=True)


def capture_image(cap, filepath, fname):

    success = False
    while (True):

        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        # quit capture
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        # save frame to file
        elif key == ord(' '):

            print("\tPress SPACE to save the frame, or another key to retry")
            key2 = cv2.waitKey(0)

            if key2 == ord(' '):
                os.makedirs(filepath, exist_ok=True)
                fullname = str(os.path.join(filepath, fname))
                cv2.imwrite(fullname, frame)
                success = True
                print(" >>> Image saved as {}".format(fullname))

            else:
                continue

    return success


def start_capture():

    while True:
        try:
            team_id = input("Enter your team number [1-7]\t> ")
            team_id = int(team_id)
            if team_id > 0 and team_id <= 7:
                break
        except:
            pass
        print("Try again")

    team_id = "T" + str(team_id)
    print(" >>> Team ID: {}".format(team_id))

    # create capture
    cap = cv2.VideoCapture(VIDEO_DEVICE)

    while True:

        obj_id = input("\n\n >>> Enter an ID for this object (alphanum and dashes only)\n\t[q]uit\t\t> ")
        if obj_id == 'q':
            break
        print(" >>> Capturing object '{}'".format(obj_id))

        filepath = os.path.join(BASE_PATH, team_id, obj_id, 'c615')
        fname = str(str(time.time()) + ".jpg")
        print(filepath)
        print(type(filepath))
        print(fname)
        print(type(fname))
        if not capture_image(cap, filepath, fname):
            break

    # release capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    if "--dirs-only" in sys.argv:
        create_dirs()
        exit()

    import cv2
    start_capture()
