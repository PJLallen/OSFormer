import cv2
import numpy as np
from PIL import Image

STEP = 8
BEZEL = 5  # pixel numbers
DURATION = 200

def img2gif(img_og, img_pred, out_name):
    im1 = cv2.imread(img_og)
    im2 = cv2.imread(img_pred)

    h, w, _ = im1.shape
    step_size = w // STEP
    res_list = []

    for i in range(STEP):
        tmp = np.ones_like(im1) * 255
        tmp[:, :(i + 1) * step_size - BEZEL] = im1[:, :(i + 1) * step_size - BEZEL]
        tmp[:, (i + 1) * step_size:] = im2[:, (i + 1) * step_size:]
        res_list.append(Image.fromarray(tmp[...,::-1].astype('uint8')).convert('RGB'))

    img = res_list[0]  # extract first image from iterator
    img.save(fp=out_name, format='GIF', append_images=res_list,
             save_all=True, duration=200, loop=0)
    

if __name__ == '__main__':
    img2gif(
        '../test_00000003.jpg',
        '../test_00000003.png',
        '../output.gif')
