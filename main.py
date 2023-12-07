import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import argparse

from calibrate import calibrate_camera
from image_helper import ImageHelper
from feature_helper import FeatureHelper

'''
Example python command: python3 main.py --data=calibration/
'''

def main(args):
    ## Class Initializations
    img_helper = ImageHelper(args['data'])
    feature_helper = FeatureHelper()
    
    ## Load Images
    img_helper.load_imgs()
    
    '''
    Perform camera calibration and get intrinsic matrix (K) and distortion coefficients)
    
    Input: 
        /filepath/to/calibration/images/ (str) - Chessboard calibration images
    Output:
        K (numpy.array) - Camera Matrix
        dist_params (numpy.array) - distortion coefficients
    '''
    print("Performing Camera Calibration...",end="\t", flush=True)
    K, dist_params, r_vecs, t_vecs = calibrate_camera('calibration') 
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
    matches = feature_helper.compute_matches(img_helper.undist_imgs)
    print("Finished!")

    '''
    Pairwise Pose Estimation

    Input: 
        - 3D feature points, ???
    Output:
        - 3D point cloud, Camera Poses
    '''


    '''
    Triangulation

    Input: 
        - 3D feature points, ???
    Output:
        - 3D point cloud, Camera Poses
    '''


    '''
    Convert to Global Frame
    
    Input: 
        - 3D feature points, ???
    Output:
        - 3D point cloud, Camera Poses
    '''

    '''
    Bundle Adjustment

    Input: 
        - 3D feature points, ???
    Output:
        - 3D point cloud, Camera Poses
    '''    

    pass

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str,help='folderpath/to/images/')
    return parser

if __name__ == '__main__':
    args = vars(read_args().parse_args())
    main(args)