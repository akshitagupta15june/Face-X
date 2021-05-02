import cv2
import os
import numpy as np
from skimage.filters import gaussian
from test import evaluate
import argparse
import imutils
import evaluate

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--img-path', default='Makeover1_Original.png')
    return parse.parse_args()

def sharpen(img):
    img = img *1.0
    gauss_out =gaussian(img, sigma=5, multichannel=True)

    alpha =1.5
    img_out= (img - gauss_out)* alpha + img

    img_out= img_out / 255.0

    mask_1 = img_out< 0
    mask_2 = img_out > 1

    img_out=img_out*(1-mask_1)
    img_out=img_out*(1-mask_2) + mask_2
    img_out= np.clip(img_out,0,1)
    img_out= img_out*255
    return np.array(img_out, dtype=np.uint8)

def hair(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r =color   #[10, 50, 250]    #[10,250,10]
    tar_color[:, :, 0]=b
    tar_color[:, :, 1]=g
    tar_color[:, :, 2]=r

    image_hsv= cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part== 12 or part== 13:
        image_hsv[:, :, 0:2]= tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1]= tar_hsv[:, :, 0:1]
    
    changed= cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part ==17:
        changed= sharpen(changed)
    
    changed[parsing !=part]= image[parsing !=part]
    return changed


if __name__ == '__main__':

    args = parse_args()

    table = {
        'hair': 17,
        'upper_lip': 12,
        'lower_lip': 13
    }

    image_path = args.image_path
    cp = 'cp/79999_iter.pth'

    image = cv2.imread(image_path)
    image = cv2.resize(image, (1024, 1024))
    ori = image.copy()
    parsing = evaluate(image_path, cp)
    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

    parts=[table['hair'], table['upper_lip'], table['lower_lip']]

    colors=[[200,0,0],[10,0,200],[10,0,200]]

    for part, color in zip(parts,colors):
        image = hair(image, parsing, part, color)
    
    cv2.imshow('image', cv2.resize(ori, (512,512)))
    cv2.imshow('color', cv2.resize(image, (512,512)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
