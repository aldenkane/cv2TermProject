# Creates directories for CV2 data collection with the following hierarchy:
# Layout > Lighting > Cameras > Views

import os
import itertools

BASE_PATH = 'collection'

dirs = {
    'layouts': [
        'l1',
        'l2',
        'l3',
    ],
    'lights': [
        'al1',
        'pl1',
        'pl2',
    ],
    'cams': [
        'c615',
        'mobi',
    ],
    'views': [
        'top',
        'side',
    ],
}

paths = list(itertools.product(dirs['layouts'], dirs['lights'], dirs['cams'], dirs['views']))
paths = [ os.path.join(*p) for p in paths ]

for p in paths:

    path = os.path.join(BASE_PATH, p)
    print("Creating {}".format(path))
    os.makedirs(path, exist_ok=True)
