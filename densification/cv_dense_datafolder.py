import os
from natsort import natsorted
import argparse
import cv2
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

def resize_img(img,ratio=1):
    new_width = int(img.shape[1] * ratio)
    new_height = int(img.shape[0] * ratio)
    img = cv2.resize(img, (new_width, new_height))
    return img

def read_args():
    parser = argparse.ArgumentParser()
    
    # Arguments
    parser.add_argument('--cam_cal',type=str,default='',help='/path/to/cam/cal.pkl')
    parser.add_argument('--img_resize_ratio',default=1.0,type=float)
    parser.add_argument('--data',type=str,help='/path/to/images/')
    return parser

if __name__ == '__main__':
    ## Load Data
    args = vars(read_args().parse_args())

    images = natsorted(os.listdir(args['data'])) # read in images
    
    # if l_img.shape[0] == 4032 or l_img.shape[0] == 403:
    #     l_img = cv2.transpose(l_img)
    #     r_img = cv2.transpose(r_img)
    cal_file = open(args['cam_cal'],'rb')
    K, dist_coeff = pkl.load(cal_file)
    cal_file.close()

    r_img = None
    l_img = None
    new_K = None
    roi = None
    # Assumes previous images is the right image (moving CW around object)
    for img_name in images:
        if r_img is None:
            r_img = cv2.imread(args['data'] + img_name, cv2.IMREAD_GRAYSCALE)
            r_img = resize_img(r_img,args['img_resize_ratio'])
            w, h = r_img.shape[1], r_img.shape[0]
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeff, (w, h), 1, (w, h))
            r_img = cv2.undistort(r_img, K, dist_coeff, None, new_K)
            roi_x,roi_y,roi_w,roi_h = roi
            r_img_crop = r_img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            r_img = cv2.resize(r_img_crop, (w,h))
            continue
        l_img = cv2.imread(args['data'] + img_name, cv2.IMREAD_GRAYSCALE)
        l_img = resize_img(l_img,args['img_resize_ratio'])
        l_img = cv2.undistort(l_img, K, dist_coeff, None, new_K)
        roi_x,roi_y,roi_w,roi_h = roi
        l_img_crop = l_img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        l_img = cv2.resize(l_img_crop, (w,h))

        ## Compute Depth Map
        # numDisparities = disparity search range, 
        # blockSize = linear size of a block (odd b/c centered at current pixel)
        #   Larger block size implies smoother, though less accurate disparity map. 
        #   Smaller block size gives more detailed disparity map, 
        #   but there is higher chance for algorithm to find a wrong correspondence.
        stereo = cv2.StereoBM_create(numDisparities=16*3, blockSize=21)
        # stereo = cv2.StereoBM_create(numDisparities=16*7, blockSize=13)#numDisparites=0, [nD=16, bS=21]
        disparity = stereo.compute(l_img,r_img)
        # disparity = disparity.astype(np.uint8)
        # disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        # print(disparity[disparity.shape[0]//2:disparity.shape[0]//2+1,:])
        # disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        fig = plt.figure()
        plt.imshow(disparity, cmap='jet')
        plt.colorbar()
        plt.axis('off')
        plt.title(f"Image {img_name}")
        plt.show()

        ## Update Previous Variables
        r_img = l_img