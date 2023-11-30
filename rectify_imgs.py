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

def get_cam_intrinsics(camera_params):
    f = camera_params["focal"]
    cx = camera_params["width"]//2
    cy = camera_params["height"]//2
    K = np.array([
        [f,0,cx],
        [0,f,cy],
        [0,0,1]
    ])
    return K

if __name__ == '__main__':
    ## Load Data
    args = vars(read_args().parse_args())
    with open(args['sfm_recon'], 'r') as file:
        reconstruction_data = json.load(file)[0]
    cal_file = open(args['cam_cal'],'rb')
    K, dist_coeff = pkl.load(cal_file)
    cal_file.close()
    # camera_params = list(reconstruction_data["cameras"].values())[0]
    # K = get_cam_intrinsics(camera_params)
    # dist_coeff = np.array([camera_params['k1'],camera_params['k2'],0,0])
    Kinv = np.linalg.inv(K)

    ## Load left & right image
    l_img = cv2.imread(args['left_img_path'])
    r_img = cv2.imread(args['right_img_path'])
    assert(l_img.shape==r_img.shape)

    ## Estimate Essential Matrix
    l_img_name = os.path.basename(args['left_img_path'])
    # l_img_name = l_img_name[12:] # remove "undistorted_" naming
    r_img_name = os.path.basename(args['right_img_path'])
    # r_img_name = r_img_name[12:] # remove "undistorted_" naming
    for img_name, img_data in reconstruction_data["shots"].items():
        if img_name == l_img_name:
            Rl, _ = cv2.Rodrigues(np.array(img_data["rotation"]))#rot_mat_from_axisangle(img_data["rotation"])
            tl = np.array(img_data["translation"])
        elif img_name == r_img_name:
            Rr, _ = cv2.Rodrigues(np.array(img_data["rotation"]))#rot_mat_from_axisangle(img_data["rotation"])
            tr = np.array(img_data["translation"])

    # Compute the rectification transformations
    R = Rr @ Rl.T # R_w2r @ (Rw2l).T # resulting in a Rotation matrix from left frame to right frame
    T = tr - tl # resulting in a translation vector from left frame to right frame

    # https://forum.opencv.org/t/unable-to-rectify-stereo-cameras-correctly/7543
    cameraMatrixL = K
    distL = dist_coeff
    widthL, heightL = l_img.shape[1], l_img.shape[0]
    cameraMatrixR = K
    distR= dist_coeff
    widthR, heightR = r_img.shape[1], r_img.shape[0]

    newCameraMatrixL, roi_L = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))
    newCameraMatrixR, roi_R = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))


    rectifyScale= 1
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv2.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, (widthL, heightL), R, T, rectifyScale,(0,0))
    # for elem in Q:
    #     print(np.round(elem))

    stereoMapL = cv2.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, (widthL, heightL), cv2.CV_16SC2)
    stereoMapR = cv2.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, (widthR, heightR), cv2.CV_16SC2)

    l_img_rect = cv2.remap(l_img, stereoMapL[0], stereoMapL[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    r_img_rect = cv2.remap(r_img, stereoMapR[0], stereoMapR[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)


    # # # Rotate Right image by applying the affine transformation to the image
    # # r_img_rot = cv2.warpAffine(r_img, R, (r_img.shape[1], r_img.shape[0]))

    # # Construct Rrect
    # print("Constructing R_rect Matrix...")
    # r1 = t / np.linalg.norm(t,2)
    # r2 = np.array([[0,-1,0],[1,0,0],[0,0,0]]) @ r1
    # r3 = np.cross(r1,r2)
    # Rrect = np.vstack((r1,r2,r3))
    # assert((3,3) == Rrect.shape)


    # # Rotate Right image by applying the affine transformation to the image
    # r_img_rect = cv2.warpPerspective(r_img, R@Rrect, (r_img.shape[1], r_img.shape[0]))
    # l_img_rect = cv2.warpPerspective(l_img, Rrect, (l_img.shape[1], l_img.shape[0]))


    # # Warp Pixels into new rectified frames
    # print("Warping Pixels into rectified coordinates...")
    # l_rect_pix = []
    # r_rect_pix = []
    # for row in range(l_img.shape[0]):
    #     for col in range(l_img.shape[1]):
    #         l_pix = K @ Rrect @ Kinv @ np.array([[col,row,1]]).T # [x,y,1]
    #         l_rect_pix.append(l_pix[:,0])
            
    #         r_pix = K @ R @ Rrect @ Kinv @ np.array([[col,row,1]]).T # [x,y,1]
    #         r_rect_pix.append(r_pix[:,0])

    # print("Warping rectified pixels into new rectified frames...")
    # l_rect_img = np.zeros_like(l_img)
    # r_rect_img = np.zeros_like(r_img)
    # idx = 0
    # for row in range(l_img.shape[0]):
    #     for col in range(l_img.shape[1]):
    #         l_pix = l_rect_pix[idx].astype(int) 
    #         if (l_pix[0] >= 0) and (l_pix[0] <= l_img.shape[1]) and \
    #             (l_pix[1] >= 0) and (l_pix[1] <= l_img.shape[0]):

    #             print(f"row: {row}, col: {col}, l_pix: {l_pix}, l_rect_img.shape: {l_rect_img.shape}, l_img.shape: {l_img.shape}")
    #             l_rect_img[row,col,:] = l_img[l_pix[1], l_pix[0], :]
            
    #         r_pix = r_rect_pix[idx].astype(int)
    #         if (r_pix[0] >= 0) and (r_pix[0] <= r_img.shape[1]) and \
    #             (r_pix[1] >= 0) and (r_pix[1] <= r_img.shape[0]):

    #             print(f"row: {row}, col: {col}, r_pix: {r_pix}, r_rect_img.shape: {r_rect_img.shape}, r_img.shape: {r_img.shape}")
    #             r_rect_img[row,col,:] = r_img[r_pix[1], r_pix[0], :]

    #         idx += 1
    
    # Display Rectified Images
    cv2.namedWindow('Left Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Left Image", l_img)
    cv2.namedWindow('Right Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Right Image", r_img)
    cv2.namedWindow('Left Rectified Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Left Rectified Image", l_img_rect)
    cv2.namedWindow('Right Rectified Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Right Rectified Image", r_img_rect)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ## Begin disparity calculations
    