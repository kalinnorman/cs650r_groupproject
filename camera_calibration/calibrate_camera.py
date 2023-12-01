import numpy as np
import cv2 
import pickle as pkl
from natsort import natsorted
import os

CHESSBOARD_SIZE = (5,7)
IMG_SIZE = (1280,720)


def save_calibration(obj_pts, img_pts):
    ret, cam_mat, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, IMG_SIZE, None, None)
    # Save calibration results
    pkl.dump((cam_mat, dist), open("calibration.pkl", "wb"))
    pkl.dump(cam_mat, open("cam_matrix.pkl", "wb"))
    pkl.dump(dist, open("dist.pkl", "wb"))
    return 

if __name__ == '__main__':
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    obj[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0],0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)

    size_of_squares_mm = 31 # mm
    obj = obj * size_of_squares_mm

    obj_pts = [] # 3D points in real world
    img_pts = [] # 2D points in image plane

    images = natsorted(os.listdir("shrunk_cal_images")) # read in images
    cv2.namedWindow('Img',cv2.WINDOW_NORMAL)
    for image_name in images:
        img = cv2.imread("shrunk_cal_images/"+str(image_name))
        # print(img.shape)
        # cv2.imshow('Img', img)
        # cv2.waitKey(1000)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        # cv2.imshow('Img', gray)
        # cv2.waitKey(1000)

        # If found, add object points and image points
        if ret == True:
            print("Found corners for image",image_name)
            obj_pts.append(obj)
            corners_subpix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            img_pts.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img,  CHESSBOARD_SIZE, corners_subpix, ret)
            cv2.imshow('Img', img)
            cv2.waitKey(1000)
        else:
            print("No corners found for image",image_name)

    cv2.destroyAllWindows()

    save_calibration(obj_pts, img_pts)

'''
-- Undistort Example --
test_img = cv2.imread(test_img.png)
h, w, c = test_img.shape
new_cam_mat, roi = cv2.getOptimialNewCameraMatrix(cam_mat, dist, (w,h), 1, (w,h))
undistort_img = cv2.undistort(test_img, cam_mat, dist, None, new_cam_mat)
x,y,w,h = roi
undistort_roi_img = undistort_img[y:y+h, x:x+w]
cv2.imwrite('undistorted_img.png', undistort_roi_img)

-- Undistort Example 2 --
test_img = cv2.imread(test_img.png)
h, w, c = test_img.shape
new_cam_mat, roi = cv2.getOptimialNewCameraMatrix(cam_mat, dist, (w,h), 1, (w,h))
mapx, mapy = cv2.initUndistortRectifyMap(cam_mat, dist, None, new_cam_mat, (w,h), 5)
undistort_img = cv2.remap(test_img, mapx, mapy, cv2.INTER_LINEAR)
x,y,w,h = roi
undistort_roi_img = undistort_img[y:y+h, x:x+w]
cv2.imwrite('undistorted_img.png', undistort_roi_img)
'''