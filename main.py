import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import argparse
import sys

from calibrate import calibrate_camera
from image_helper import ImageHelper
from feature_helper import FeatureHelper
from bundle_adjustment import BundleAdjustment

'''
Example python command: python3 main.py --data=calibration/
'''

def main(args):
    ## Class Initializations
    img_helper = ImageHelper(args['data'])
    # img_helper = ImageHelper('imgs/sfm/')
    feature_helper = FeatureHelper()
    bundle_adjustment = BundleAdjustment()
    
    ## Load Images
    print("Loading Images...",end="\t", flush=True)
    img_helper.load_imgs()
    print("Finished!")

    '''
    Perform camera calibration and get intrinsic matrix (K) and distortion coefficients)
    
    Input: 
        /filepath/to/calibration/images/ (str) - Chessboard calibration images
    Output:
        K (numpy.array) - Camera Matrix
        dist_params (numpy.array) - distortion coefficients
    '''
    print("Performing Camera Calibration...",end="\t", flush=True)
    K, dist_params, r_vecs, t_vecs = calibrate_camera('imgs/calibration') 
    print("Finished!") 

    '''
    Undistort Images

    Input:
        K (np.array) - intrinsic camera calibration matrix
        dist_params (np.array) - distortion coefficients
    '''
    print("Undistorting Images...",end="\t", flush=True)
    img_helper.undistort_imgs(K, dist_params, display=False)
    print("Finished!")

    '''
    Feature Matching:
    Performs feature matching between adjacent images.
    Assumes images are named sequentially. 

    Input: 
        - undistorted images (dict):
            key = img_name,
            value = img
    Output:
        - matches (dict): 
            key = (img_name1, img_name2), 
            value = [(key_point1, key_point2), ...] 
    '''
    print("Performing Feature Matching...",end="\t", flush=True)
    # feature_helper.compute_matches(img_helper.undist_imgs)
    feature_helper.compute_matches_alt(img_helper.undist_imgs)
    print("Finished!")

    '''
    Initial Pose and 3D Point (Triangulation) Estimation

    Input: 
        - Camera Intrinsic Matrix (This assumes a single camera for all images)
    Output:
        - Rotation, unit Translation, 3D point cloud
    '''
    print("Performing Initial Pose and 3D Point Estimation...",end="\t", flush=True)
    Rs, ts, pts_3d, helper_list = feature_helper.estimate_pairwise_poses_and_3d_points(K)
    print("Finished!")

    '''
    Bundle Adjustment

    Input: 
        - Camera Intrinsic Matrix, Initial Rotation, Initial Translation, 3D Point List, and Feature Matching List
    Output:
        - Updated Rotation, Translation, and 3D point estimates
    '''    
    print("Performing Bundle Adjustment...",end="\t", flush=True)
    Rs, ts, pts_3d = bundle_adjustment.run(K, Rs, ts, pts_3d, helper_list)
    print("Finished!")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts_3d[:,0], pts_3d[:,1], pts_3d[:,2])
    plt.show()

    pass

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str,help='folderpath/to/images/')
    return parser

if __name__ == '__main__':
    args = vars(read_args().parse_args())
    main(args)