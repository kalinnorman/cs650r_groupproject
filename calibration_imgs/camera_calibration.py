import numpy as np
import cv2 as cv
import os

# Get list of filenames for images in calibration_imgs folder
imgs_filepath = '/'.join(__file__.split('/')[:-1])
imgs_filenames = os.listdir(imgs_filepath)
imgs_filenames = [i for i in imgs_filenames if '.dng' in i] # Only keep .dng files
imgs_filenames.sort()

# Checkerboard information
dim = (10, 7)

# Prepare object points (these are the 3D world coordinates of the checkerboard corners, where we assume z=0, therefore the checkerboard is flat in the xy plane)
objp = np.zeros((dim[0]*dim[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:dim[0], 0:dim[1]].T.reshape(-1, 2)

# Arrays to store object and image points from all images
objpts = [] # 3D points in real world space
imgpts = [] # 2D points in image plane

# How each image should (or shouln't) be rotated
rotations = [0, 0, 0, 0, 0, 0, 3, 2, 1, 1, 3, 3, 0, 0, 
             0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 3, 3, 3, 
             0, 3, 1, 1, 1, 1, 2, 3, 3, 3, 3, 0, 0, 0, 
             0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 
             3, 2, 1, 3, 3, 3, 0, 0, 1, 1, 1, 0, 3, 2]

# Iterate through images and track number of successful checkerboard detections
success_count = 0
for i, filename in enumerate(imgs_filenames):
    # Open image
    img_path = os.path.join(imgs_filepath, filename)
    img = cv.imread(img_path)
    # Rotate if necessary
    if rotations[i] == 1:
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    elif rotations[i] == 2:
        img = cv.rotate(img, cv.ROTATE_180)
    elif rotations[i] == 3:
        img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find checkerboard corners
    ret, corners = cv.findChessboardCorners(gray, dim, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        success_count += 1
        objpts.append(objp)
        # Refine corners
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpts.append(corners2)
        # # Draw and display the corners
        # cv.drawChessboardCorners(img, dim, corners2, ret)
        # cv.imshow('img', img)
        # if cv.waitKey(0) == ord('q'):
        #     cv.destroyAllWindows()
        #     break
        # cv.destroyAllWindows()
    
# Print number of successful images
print(f'Number of successful images: {success_count} / {len(imgs_filenames)}')

# Calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpts, imgpts, gray.shape[::-1], None, None)

# Print camera matrix and distortion coefficients
print(f'Camera matrix: {mtx}')
print(f'Distortion coefficients: {dist}')

# Reprojection error
mean_error = 0
for i in range(len(objpts)):
    imgpoints2, _ = cv.projectPoints(objpts[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpts[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print("Reprojection error: {}".format(mean_error/len(objpts)) )