import sys
import ctypes
import numpy as np
from PIL import Image
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2 as cv
import glob
import os
np.set_printoptions(suppress=True, edgeitems=30, linewidth=256)


def show_img(img, caption):
    f, axarr = plt.subplots(1,1, figsize=(16,16))
    axarr.imshow(img)
    axarr.set_axis_off()
    axarr.set_title(caption, y=-.04)
    
def calibrate_camera(path):
    cwd = os.getcwd()
    path = os.path.join(cwd, path)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    path_search = os.path.join(path, '*.jpg')
    images = glob.glob(path_search)
    print(images)

    # Step through the list and search for chessboard corners
    w = None
    h = None
    for idx, fname in enumerate(images):
        img = cv.imread(fname)
        if h is not None and w is not None:
            assert w == img.shape[1] and h == img.shape[0], 'Images cannot be of inconsistent shape'
        else:
            w = img.shape[1]
            h = img.shape[0]
            
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, (8,6), None)

        # If found, add object points, image points
        if ret == True:
            corners = np.array(corners)
            corners = corners.reshape(corners.shape[0], corners.shape[-1])
            
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            # cv.drawChessboardCorners(img, (8,6), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            # cv.imshow('img', img)
            # cv.waitKey(500)
        print('Done with ' + str(idx))

    # cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (w,h), None, None)

    return mtx, dist, rvecs, tvecs
    


