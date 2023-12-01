import argparse
import cv2
import os
from natsort import natsorted

def resize_img(img,ratio=1):
    new_width = int(img.shape[1] * ratio)
    new_height = int(img.shape[0] * ratio)
    img = cv2.resize(img, (new_width, new_height))
    return img

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folderpath',type=str,help='/path/to/data/images')
    return parser

if __name__ == '__main__':
    ## Load Data
    args = vars(read_args().parse_args())

    img_list = natsorted(os.listdir(args['img_folderpath']+"/"))
    img_ratio = 0.1
    dir = os.path.dirname(args['img_folderpath'])
    dir += "/shrunk_images/"
    os.makedirs(dir, exist_ok=True)
    for img_name in img_list:
        resized_img = resize_img(cv2.imread(args['img_folderpath']+"/" + img_name), img_ratio) 
        cv2.imwrite(dir + img_name, resized_img)