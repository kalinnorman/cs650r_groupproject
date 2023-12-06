import numpy as np
import matplotlib.pyplot as plt

from calibrate import calibrate_camera

def main():
    
    '''
    Perform camera calibration and get intrinsic matrix (K) and distortion coefficients)
    
    Input: 
        /filepath/to/calibration/images/ (str) - Chessboard calibration images
    Output:
        K (numpy.array) - Camera Matrix
        dist_params (numpy.array) - distortion coefficients
    '''
    K, dist_params, r_vecs, t_vecs = calibrate_camera('calibration') 
    
    '''
    Read in images & Undistort them

    Input: 
        /filepath/to/images/ (str) - to image dataset folder
        K (np.array) - intrinsic camera calibration matrix
        dist_params (np.array) - distortion coefficients
    Output:
        undistorted_imgs (List) - 
    '''
    # undistorted_imgs = load_and_preprocess_imgs()

    '''
    Feature Matching

    Input: 
        - 3D feature points, ???
    Output:
        - 3D point cloud, Camera Poses
    '''


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

if __name__ == '__main__':
    main()