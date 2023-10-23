import os
import argparse
import pickle as pkl
from natsort import natsorted
import cv2
import numpy as np

def undistort_img(img,cam_mat,dist_coeffs):
    h, w = img.shape
    new_cam_mat, roi = cv2.getOptimalNewCameraMatrix(cam_mat, dist_coeffs, (w,h), 1, (w,h))
    undistorted_img = cv2.undistort(img, cam_mat, dist_coeffs, None, new_cam_mat)
    x,y,w,h = roi
    undistorted_roi_img = undistorted_img[y:y+h,x:x+w]
    corrected_img = cv2.resize(undistorted_roi_img,img.shape)
    return corrected_img

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_data',help='/path/to/data/imgs/')
    parser.add_argument('--cam_cal',help='/path/to/camera/calibration/parameters.pkl')  
    return parser

if __name__ == '__main__':
    args = vars(read_args().parse_args())
    
    # Get Camera Calibration params
    cal_file = open(args['cam_cal'],'rb')
    cam_mat, dist_coeff = pkl.load(cal_file)
    cal_file.close()

    # Initiate SIFT detector AND Brute-Force Matcher
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    # Read in images
    prev_img = None
    curr_img = None
    img_dir = natsorted(os.listdir(args['img_data']))
    cv2.namedWindow('Matches img', cv2.WINDOW_NORMAL)
    
    for i, img_name in enumerate(img_dir):
        if i == 0:
            img = cv2.imread(args['img_data'] + img_name,0)
            # cv2.imshow("Matches img", img)
            # cv2.waitKey(0)
            prev_img = undistort_img(img,cam_mat, dist_coeff)
            # cv2.imshow("Matches img", prev_img)
            # cv2.waitKey(0)
            continue
        
        img = cv2.imread(args['img_data'] + img_name,0)
        curr_img = undistort_img(img, cam_mat, dist_coeff)

        # Extract keypoints and descriptors from both images
        kp_p, des_p = sift.detectAndCompute(prev_img, None)
        kp_c, des_c = sift.detectAndCompute(curr_img, None)

        # Get matches
        matches = bf.knnMatch(des_p,des_c,k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        
        # Display matches
        match_img = cv2.drawMatchesKnn(
            prev_img,
            kp_p,
            curr_img,
            kp_c,
            good,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        prev_img = np.copy(curr_img)

        cv2.imshow("Matches img", match_img)
        cv2.waitKey(1000)

    cv2.destroyAllWindows()
