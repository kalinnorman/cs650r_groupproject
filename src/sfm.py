import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os

from triangulation import triangulate

'''
NOTE : This file is written in, and for, linux systems.
'''

# User Inputs
img_folder_name = 'test_imgs'
img_filetype = '.dng'
intrinsic_matrix = np.array([[925.79882927,   0.,         635.51907178], # Intrinsic matrix (from camera calibration)
                             [  0.,         923.71342657, 483.87251378],
                             [  0.,           0.,           1.        ]])
distortion_coeffs = np.array([[0.0937379544, -0.360357661, 0.000903468189, 0.000267717772, 0.589424854]])

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

# Create Brute Force Matcher
bf = cv.BFMatcher(normType=cv.NORM_L2, crossCheck=True)

# Variables to hold previous variables
prev_bgr_img = None
prev_gray_img = None
prev_kp, prev_des = None, None

# Find feature points in each image
for i in range(num_imgs):
    # Read image
    img_filepath = os.path.join(imgs_dir, img_names[i])
    bgr_img = cv.imread(img_filepath)
    bgr_img = cv.undistort(bgr_img, intrinsic_matrix, distortion_coeffs)
    # cv.imshow('img', bgr_img)
    # cv.waitKey(0)
    # exit()
    # Convert to grayscale
    gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
    # Find keypoints and descriptors
    kp, des = sift.detectAndCompute(gray_img, None)
    
    # # (OPTIONAL) Draw keypoints on image
    # kp_img = cv.drawKeypoints(gray_img, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv.imshow('Keypoints', kp_img)
    # cv.waitKey(0)
    
    # Match keypoints with previous image
    if prev_kp is not None:
        # Identify matches
        matches = bf.match(des, prev_des) # the match function takes in (query descriptors, train descriptors)
        matches = sorted(matches, key=lambda x:x.distance)
        # Pull out points from matches
        pts_prev = []
        pts_curr = []
        for i, match in enumerate(matches):
            pts_prev.append(prev_kp[match.trainIdx].pt)
            pts_curr.append(kp[match.queryIdx].pt)
        # Convert points from list to array
        pts_prev = np.array(pts_prev)
        pts_curr = np.array(pts_curr)

        # # (OPTIONAL) Plot the matches
        # match_img = cv.drawMatches(gray_img, kp, prev_gray_img, prev_kp, matches, None)
        # cv.imshow('Matches', match_img)
        # if cv.waitKey(0) == ord('q'):
        #     break
        
        # Estimate the fundamental matrix
        F, mask = cv.findFundamentalMat(pts_prev, pts_curr, cv.FM_RANSAC)
        # Identify inlier points
        pts_prev_inlier = pts_prev[mask.ravel()==1]
        pts_curr_inlier = pts_curr[mask.ravel()==1]
        
        # (OPTIONAL) Find and plot the epilines
        # lines_prev = cv.computeCorrespondEpilines(pts_curr.reshape(-1,1,2), 2, F)
        # lines_prev = lines_prev.reshape(-1,3)
        # lines_curr = cv.computeCorrespondEpilines(pts_prev.reshape(-1,1,2), 1, F)
        # lines_curr = lines_curr.reshape(-1,3)
        # epilines_img_prev_bgr = prev_bgr_img.copy()
        # r, c = epilines_img_prev_bgr.shape[:2]
        # for r, pt1, pt2 in zip(lines_prev, pts_prev, pts_curr):
        #     color = tuple(np.random.randint(0,255,3).tolist())
        #     x0, y0 = map(int, [0, -r[2]/r[1]])
        #     x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        #     img_temp = cv.line(epilines_img_prev_bgr, (x0,y0), (x1,y1), color, 1)
        #     img_temp = cv.circle(img_temp, np.int32(pt1), 5, color, -1)
        # cv.imshow('Epilines', img_temp)
        # if cv.waitKey(0) == ord('q'):
        #     break
        # epilines_img_curr_bgr = bgr_img.copy()
        # r, c = epilines_img_curr_bgr.shape[:2]
        # for r, pt1, pt2 in zip(lines_curr, pts_curr, pts_prev):
        #     color = tuple(np.random.randint(0,255,3).tolist())
        #     x0, y0 = map(int, [0, -r[2]/r[1]])
        #     x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        #     img_temp = cv.line(epilines_img_curr_bgr, (x0,y0), (x1,y1), color, 1)
        #     img_temp = cv.circle(img_temp, np.int32(pt1), 5, color, -1)
        # cv.imshow('Epilines', img_temp)
        # if cv.waitKey(0) == ord('q'):
        #     break
        
        # Estimate the essential matrix
        E = intrinsic_matrix.T @ F @ intrinsic_matrix
        # Estimate the camera pose from the essential matrix
        retval, R, t, mask = cv.recoverPose(E, pts_prev_inlier, pts_curr_inlier, intrinsic_matrix)
        # Construct projection matrices
        P_prev = intrinsic_matrix @ np.hstack((np.eye(3), np.zeros((3,1))))
        P_curr = intrinsic_matrix @ np.hstack((R, t))
        # Triangulate to estimate 3D points
        pts_3d_homogenous = cv.triangulatePoints(P_prev, P_curr, pts_prev_inlier.T, pts_curr_inlier.T)
        pts_3d = (pts_3d_homogenous[:3, :] / pts_3d_homogenous[3,:]).T # N x 3 array
        pts_3d_alt = triangulate(intrinsic_matrix, E, R, pts_prev_inlier, pts_curr_inlier)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts_3d[:,0], pts_3d[:,1], pts_3d[:,2])
        # ax.scatter(pts_3d_alt[:,0], pts_3d_alt[:,1], pts_3d_alt[:,2])
        plt.show()
        exit()
    # Update previous variables
    prev_bgr_img = bgr_img.copy()
    prev_gray_img = gray_img.copy()
    prev_kp = kp
    prev_des = des
    
        
