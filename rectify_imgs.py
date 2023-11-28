import os
import json
import argparse
import numpy as np
import pickle as pkl
import cv2


def rot_mat_from_axisangle(axis_angle):
    x,y,z = axis_angle / np.linalg.norm(axis_angle,2)
    S = np.array([
        [0,-z,y],
        [z,0,-x],
        [-y,x,0]
    ])
    theta = np.linalg.norm(axis_angle,2)
    R = np.eye(3) + \
        np.sin(theta) * S + \
        (1 - np.cos(theta)) * S**2
    return R

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_recon',type=str,help='/path/to/reconstruction.json')
    
    parser.add_argument('--cam_cal',help='/path/to/camera/calibration/parameters.pkl')
    
    parser.add_argument('--left_img_path',type=str,default="l",help='left image path name (as appears in reconstruction file')
    parser.add_argument('--right_img_path',type=str,default="r",help='right image path name (as appears in reconstruction file')  
    return parser

if __name__ == '__main__':
    ## Load Data
    args = vars(read_args().parse_args())
    with open(args['sfm_recon'], 'r') as file:
        reconstruction_data = json.load(file)[0]
    cal_file = open(args['cam_cal'],'rb')
    K, dist_coeff = pkl.load(cal_file)
    Kinv = np.linalg.inv(K)
    cal_file.close()

    ## Load left & right image
    l_img = cv2.imread(args['left_img_path'])
    r_img = cv2.imread(args['right_img_path'])
    assert(l_img.shape==r_img.shape)

    ## Estimate Essential Matrix
    l_img_name = os.path.basename(args['left_img_path'])
    r_img_name = os.path.basename(args['right_img_path'])
    for img_name, img_data in reconstruction_data["shots"].items():
        if img_name == l_img_name:
            Rl = rot_mat_from_axisangle(img_data["rotation"])
            tl = np.array(img_data["translation"])
        elif img_name == r_img_name:
            Rr = rot_mat_from_axisangle(img_data["rotation"])
            tr = np.array(img_data["translation"])
    R = Rr - Rl
    t = tr - tl

    # Construct Rrect
    print("Constructing R_rect Matrix...")
    r1 = t / np.linalg.norm(t,2)
    r2 = np.array([[0,-1,0],[1,0,0],[0,0,0]]) @ r1
    r3 = np.cross(r1,r2)
    Rrect = np.vstack((r1,r2,r3))
    assert((3,3) == Rrect.shape)

    # Warp Pixels into new rectified frames
    print("Warping Pixels into rectified coordinates...")
    l_rect_pix = []
    r_rect_pix = []
    for row in range(l_img.shape[0]):
        for col in range(l_img.shape[1]):
            l_pix = K @ Rrect @ Kinv @ np.array([[col,row,1]]).T # [x,y,1]
            l_rect_pix.append(l_pix[:,0])
            
            r_pix = K @ R @ Rrect @ Kinv @ np.array([[col,row,1]]).T # [x,y,1]
            r_rect_pix.append(r_pix[:,0])

    print("Warping rectified pixels into new rectified frames...")
    l_rect_img = np.zeros_like(l_img)
    r_rect_img = np.zeros_like(r_img)
    idx = 0
    for row in range(l_img.shape[0]):
        for col in range(l_img.shape[1]):
            l_pix = l_rect_pix[idx].astype(int) 
            if (l_pix[0] >= 0) and (l_pix[0] <= l_img.shape[1]) and \
                (l_pix[1] >= 0) and (l_pix[1] <= l_img.shape[0]):

                # print(f"row: {row}, col: {col}, l_pix: {l_pix}, l_rect_img.shape: {l_rect_img.shape}, l_img.shape: {l_img.shape}")
                l_rect_img[row,col,:] = l_img[l_pix[1], l_pix[0], :]
            
            r_pix = r_rect_pix[idx].astype(int)
            if (r_pix[0] >= 0) and (r_pix[0] <= r_img.shape[1]) and \
                (r_pix[1] >= 0) and (r_pix[1] <= r_img.shape[0]):

                # print(f"row: {row}, col: {col}, r_pix: {r_pix}, r_rect_img.shape: {r_rect_img.shape}, r_img.shape: {r_img.shape}")
                r_rect_img[row,col,:] = r_img[r_pix[1], r_pix[0], :]

            idx += 1
    
    # Display Rectified Images
    cv2.namedWindow('Left Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Left Image", l_img)
    cv2.namedWindow('Right Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Right Image", r_img)
    cv2.namedWindow('Left Rectified Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Left Rectified Image", l_rect_img)
    cv2.namedWindow('Right Rectified Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Right Rectified Image", r_rect_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()