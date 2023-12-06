import numpy as np
import matplotlib.pyplot as plt

def main():
    
    '''
    Perform camera calibration and get intrinsic matrix (K) and distortion coefficients)
    '''
    K, dist_params = calibrate_camera() 

    '''
    Read in images
    '''
    undistorted_imgs = load_and_preprocess_imgs()

    '''
    Feature Matching
    '''


    '''
    Pairwise Pose Estimation
    '''


    '''
    Triangulation
    '''


    '''
    Convert to Global Frame
    '''

    
    
    

    pass

if __name__ == '__main__':
    main()