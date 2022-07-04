import json
import os
import cv2
import numpy as np


BASEDIR = 'SOTA'
DATASETS = {
    'cod': {'image': 'COD10K-v3/Test/Image/',
            'gt': 'COD10K-v3/Test/GT_Instance/'},
    'nc4k': {'image': 'NC4K/Imgs/',
             'gt': 'NC4K/Instance/'}}
ORDER_JSON = 'SOTA/desc_res_{}.json'
OUTPUT_DIR = 'Combined'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# 4 rows 3 columns
def combine_a_image(filename, dataset, score, order, nums_per_row=3, nums_per_col=4):
    print(filename)
    gt = cv2.imread(os.path.join(DATASETS[dataset]['gt'], filename.replace('.jpg', '.png')))
    cv2.putText(gt, score, (60, 60), cv2.FONT_HERSHEY_PLAIN, 5.0, (0, 0, 255), 4)
    shape = gt.shape
    im = cv2.imread(os.path.join(DATASETS[dataset]['image'], filename))
    target_shape = [shape[0] * nums_per_col, shape[1] * nums_per_row, 3]
    result = np.zeros(target_shape)
    result[:shape[0], :shape[1], :] = gt

    i = 0
    for method in os.listdir(BASEDIR):
        if not os.path.isdir(os.path.join(BASEDIR, method)):
            continue

        print(method)
        i += 1
        row = i // nums_per_row
        col = i % nums_per_row

        vis_map = cv2.imread(os.path.join(BASEDIR, method, 'vis',  filename))
        if vis_map is None:
            vis_map = im.copy()
        if vis_map.shape != shape:
            vis_map = cv2.resize(vis_map, (shape[1], shape[0]))
        cv2.putText(vis_map, method, (60, 60), cv2.FONT_HERSHEY_PLAIN, 5.0, (0, 0, 255), 4)
        result[row * shape[0]: (row + 1) * shape[0],
               col * shape[1]: (col + 1) * shape[1], :] = vis_map
    cv2.imwrite(os.path.join(OUTPUT_DIR, order + filename), result)
    print('Save {} successfully!'.format(os.path.join(OUTPUT_DIR, order + filename)))


for dataset_ in ['nc4k']: # DATASETS.keys():
    with open(ORDER_JSON.format(dataset_), 'r') as f:
        order_dict = json.load(f)

    for idx, (filename_, score_delta) in enumerate(order_dict.items()):
        score_fmt = '{:.2f}'.format(score_delta * 100)
        combine_a_image(filename_, dataset_, score_fmt, '{:04}_'.format(idx))
