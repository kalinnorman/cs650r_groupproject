import numpy as np
import cv2 as cv
import os

'''
NOTE : This file is written in, and for, linux systems.
'''

# User Inputs
img_folder_name = 'test_imgs'
img_filetype = '.dng'
intrinsic_matrix = np.array([[925.79882927,   0.,         635.51907178], # Intrinsic matrix (from camera calibration)
                             [  0.,         923.71342657, 483.87251378],
                             [  0.,           0.,           1.        ]])

# Useful directories
file_dir = '/'.join(__file__.split('/')[:-1])
repo_dir = '/'.join(file_dir.split('/')[:-1])
imgs_dir = os.path.join(repo_dir, img_folder_name)

# Identify and sort all image filenames in the image directory
img_names = [img for img in os.listdir(imgs_dir) if img.endswith(img_filetype)]
img_names.sort()
num_imgs = len(img_names)

# Create SIFT feature detector
sift = cv.SIFT_create()

# Find feature points in each image
for i in range(num_imgs):
    # Read image
    img_filepath = os.path.join(imgs_dir, img_names[i])
    bgr_img = cv.imread(img_filepath)

    cv.imshow('img', bgr_img)
    cv.waitKey(0)
    # Convert to grayscale
    gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)

