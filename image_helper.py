import os
import cv2 as cv
import numpy as np
from natsort import natsorted

class ImageHelper():
    def __init__(self, img_folderpath):
        self.img_folderpath = img_folderpath
        self.img_h = None
        self.img_w = None
        self.orig_imgs = dict()
        self.undist_imgs = dict()
        return
    
    def undistort_imgs(self, cam_mat, dist_coeff, display=False):
        new_cam_mat, roi = cv.getOptimalNewCameraMatrix(cam_mat,
                                                        dist_coeff,
                                                        (self.img_w,self.img_h),
                                                        1,
                                                        (self.img_w,self.img_h))
        x,y,w,h = roi
        for img_name, img in self.orig_imgs.items():
            undist_img = cv.undistort(img, cam_mat, dist_coeff, None, new_cam_mat)
            undist_img_roi = undist_img[y:y+h, x:x+w, :]
            self.undist_imgs[img_name] = cv.resize(undist_img_roi, (self.img_w,self.img_h))
        if display:
            first_img_name = next(iter(self.orig_imgs))
            cv.imshow('Distorted Image',self.orig_imgs[first_img_name])
            cv.imshow('Undistorted Image',self.undist_imgs[first_img_name])
            cv.waitKey(0)
            cv.destroyAllWindows()
        return #self.undist_imgs
    
    def load_imgs(self, img_limit=None):
        image_names = natsorted(os.listdir(self.img_folderpath))
        for img_name in image_names:
            img = cv.imread(self.img_folderpath + img_name)
            if self.img_h is None:
                self.img_h, self.img_w = img.shape[0], img.shape[1]
            self.orig_imgs[img_name] = img
            if img_limit is not None and len(self.orig_imgs) >= img_limit:
                break
        return
