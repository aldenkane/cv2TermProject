import augmentations as ag
import os
import cv2
from PIL import Image

rootdir ='./'
for subdir, dirs, files in os.walk(rootdir):
    for filei in files:
        if(filei.endswith('.jpg')):
            print(os.path.join(subdir,filei), filei)
            img=Image.open(os.path.join(subdir,filei))
            outimg=ag.Solarize(img,128)
            light=ag.Lighting(1,1,1)
            outimg=light.__call__(img)
            img.show()
            outimg.show()
            raise Exception('Only One Image at a time')

        
