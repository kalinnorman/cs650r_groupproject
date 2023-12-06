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
np.set_printoptions(suppress=True, edgeitems=30, linewidth=256)


def show_img(img, caption):
    f, axarr = plt.subplots(1,1, figsize=(16,16))
    axarr.imshow(img)
    axarr.set_axis_off()
    axarr.set_title(caption, y=-.04)
    
def sRGB_to_grayscale_linear(img):
    
    # Normalize to 0.0->1.0 range
    n_img = np.array(img, dtype=np.float32) / 255.0

    # Gamma correction removal - see https://en.wikipedia.org/wiki/Grayscale
    lower_mask   = n_img <= 0.04045
    upper_mask   = n_img >  0.04045
    lower_update = n_img[:] / 12.92
    upper_update = np.power( ( (n_img[:] + 0.055) / 1.055), 2.4 )
    
    l_rgb = (lower_mask * lower_update) + (upper_mask * upper_update)  
    
    # Map to RGB
    gray = l_rgb[:,:,0] * 0.2126 + l_rgb[:,:,1] * 0.7152 + l_rgb[:,:,2] * 0.0722     
    return gray
    
img_test = np.asarray(Image.open('./calibration/GOPR0034.jpg'))
show_img(img_test, 'TESTING')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('.\\calibration\\*.jpg')
print(images)

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv.imread(fname)
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

cv.destroyAllWindows()



# OPENCV way
img = cv.imread('.\\calibration\\GOPR0034.jpg')
h, w, z = img.shape[:]
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (w,h), None, None)

print('Reference Camera_Intrinsics:\n', mtx)
print('Reference Camera_Distortion:\n', dist)
#newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

dst = cv.undistort(img, mtx, dist, None, mtx)
dst_small = cv.resize(dst, ( int(dst.shape[1]), int(dst.shape[0])) )
show_img(dst_small, 'Reference Undistortion')
cv.imshow('undist', dst_small)
cv.waitKey()