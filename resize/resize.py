import os
import json
import cv2
import numpy as np
from copy import deepcopy

multi_img_name = 'l1_al1_m_top_5' #use a tote image to find relative scale
single_img_names = os.listdir('masks')

with open('masks/' + multi_img_name + '.json', 'r') as f:
    multi_mask = json.load(f)
multi_img = cv2.imread('images/' + multi_img_name + '.jpg')

for single_img_name in single_img_names:

    with open('masks/' + single_img_name, 'r') as f:
        single_mask = json.load(f)
    
    if len(single_mask['shapes']) != 1:
        continue
    single_img_name = single_img_name.rsplit('.', 1)[0] 
    # NOTE: imgs are (w, h), but masks are (h, w)

    single_img = cv2.imread('images/' + single_img_name + '.jpg')
    if single_img is None:
        print(single_img_name, 'WAS NOT FOUND')
        continue

    print(single_img_name)
    def minmax_wh(poly):
        wmin = hmin = float('+inf')
        wmax = hmax = float('-inf')
        for (h, w) in poly:
            wmin = min(wmin, w)
            wmax = max(wmax, w)
            hmin = min(hmin, h)
            hmax = max(hmax, h)
        return wmax - wmin, hmax - hmin


    assert len(single_mask['shapes']) == 1
    single_label = single_mask['shapes'][0]['label']
    single_poly = single_mask['shapes'][0]['points']
    single_wh = minmax_wh(single_poly)  # width/height of object in single image

    for shape in multi_mask['shapes']:
        if shape['label'] == single_label:
            multi_poly = shape['points']
            multi_wh = minmax_wh(
                multi_poly)  # width/height of object in multi image
            break
    else:
        raise RuntimeError(multi_img_name + ' does not contain ' + single_label)

    # time to resize
    scale = single_wh[0] / multi_wh[0]
    scale = max(scale, single_wh[1] / multi_wh[1])
    if scale < 1:
        print(single_img_name, 'SCALE LESS THAN 1', scale, '- continuing,,,')
        continue
    # assert scale == single_wh[1] / multi_wh[1]
    wnew = round(single_img.shape[0] / scale)
    hnew = round(single_img.shape[1] / scale)
    print('SCALES', single_wh[0] / multi_wh[0],  single_wh[1] / multi_wh[1])

    single_img_resized = cv2.resize(single_img, (hnew, wnew))

    single_mask_new = deepcopy(single_mask)
    hmid = (multi_img.shape[1] - hnew) // 2
    wmid = (multi_img.shape[0] - wnew) // 2
    single_mask_new['shapes'][0]['points'] = [
        (h / scale + hmid, w / scale + wmid)
        for h, w in single_mask_new['shapes'][0]['points']]

    single_mask_new['imageHeight'] = multi_mask['imageHeight']
    single_mask_new['imageWidth'] = multi_mask['imageWidth']
    del single_mask_new['imageData']

    if False:
        single_img_new = np.zeros_like(single_img)

        single_img_new[wmid:wmid + wnew, hmid:hmid + hnew] = single_img_resized
        # print(single_img.shape)
    else:
        #single_img_new = cv2.copyMakeBorder(single_img_resized,
                                            #wmid, wmid, hmid, hmid,
                                            #cv2.BORDER_REPLICATE)
        single_img_new = cv2.copyMakeBorder(single_img_resized,
                                            wmid, wmid, hmid, hmid,
                                            cv2.BORDER_CONSTANT,value=(220,220,220))
                                                                                
    if not os.path.exists('masks_new'):
        os.mkdir('masks_new')
    with open('masks_new/' + single_img_name + '_white.json', 'w') as f:
        json.dump(single_mask_new, f)

    if not os.path.exists('images_new'):
        os.mkdir('images_new')
    cv2.imwrite('images_new/' + single_img_name + '_white.jpg', single_img_new)
    '''
    if True:
        import matplotlib.pyplot as plt

        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(single_img_new)
        ax2.imshow(multi_img)
        x, y = zip(*single_mask_new['shapes'][0]['points'])
        ax1.scatter(x, y)
        x, y = zip(*multi_poly)
        ax2.scatter(x, y)
        plt.show()
        '''
