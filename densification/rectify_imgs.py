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

def calc_disparity(l_img,r_img,row_idx,col_idx,patch_sz,d_max=None):
    '''
    Compute disparity at a given patch in the left image
    Return: disparity, right_img_col_idx
    '''
    if (d_max is None) or (int(col_idx - d_max) <= 0):
        start_idx = 0
    else:
        if d_max < patch_sz:
            start_idx = int(col_idx - patch_sz//2)
        else:
            start_idx = int(col_idx - d_max)
    # print(patch_sz, col_idx)
    scan_steps = np.arange(start_idx + patch_sz//2, col_idx + 1, 1)#patch_sz)
    if len(scan_steps) == 1:
        return 0
    ssd = np.zeros((len(scan_steps),))
    # print(scan_steps, ssd)
    
    l_img_patch = l_img[row_idx - patch_sz//2 : row_idx + np.ceil(patch_sz/2).astype(int), 
                        col_idx - patch_sz//2 : col_idx + np.ceil(patch_sz/2).astype(int), :]
    wL_b = l_img_patch[:,:,0].reshape((patch_sz*patch_sz,))
    wL_g = l_img_patch[:,:,1].reshape((patch_sz*patch_sz,))
    wL_r = l_img_patch[:,:,2].reshape((patch_sz*patch_sz,))
    cnt = 0
    for col_scanline in range(start_idx + patch_sz//2, col_idx + 1, 1):#patch_sz):
        d = col_idx - col_scanline
        r_img_patch = r_img[row_idx - patch_sz//2 : row_idx + np.ceil(patch_sz/2).astype(int), 
                            col_scanline - patch_sz//2 : col_scanline + np.ceil(patch_sz/2).astype(int), :]
        wR_b = r_img_patch[:,:,0].reshape((patch_sz*patch_sz,))
        ssd[cnt] += np.linalg.norm(wL_b - wR_b,2)**2
        wR_g = r_img_patch[:,:,1].reshape((patch_sz*patch_sz,))
        ssd[cnt] += np.linalg.norm(wL_g - wR_g,2)**2     
        wR_r = r_img_patch[:,:,2].reshape((patch_sz*patch_sz,))
        ssd[cnt] += np.linalg.norm(wL_r - wR_r,2)**2
        ssd[cnt] /= 3 # normalize per channel
        cnt += 1
    # print(ssd)
    wta_idx = np.argmax(ssd) # winner-takes-all
    disp = col_idx - scan_steps[wta_idx]
    # print("Disparity:", disp)
    return disp


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
    cal_file.close()
    # camera_params = list(reconstruction_data["cameras"].values())[0]
    # K = get_cam_intrinsics(camera_params)
    # dist_coeff = np.array([camera_params['k1'],camera_params['k2'],0,0])
    Kinv = np.linalg.inv(K)

    ## Load left & right image
    l_img = cv2.imread(args['left_img_path'])
    r_img = cv2.imread(args['right_img_path'])
    assert(l_img.shape==r_img.shape)
    ratio = 1
    new_width = int(l_img.shape[1] * ratio)
    new_height = int(l_img.shape[0] * ratio)
    # Resize the image
    l_img = cv2.resize(l_img, (new_width, new_height))
    r_img = cv2.resize(r_img, (new_width, new_height))

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
    R = Rr @ Rl.T # R_w2r @ (R_w2l).T # resulting in a Rotation matrix from left frame to right frame
    T = tr - tl # resulting in a translation vector from left frame to right frame

    print("Rectifying Images...")
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
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

    ## Begin disparity calculations
    # Check that the horizontal row between images are the same
    # Display Rectified Images
    nrows = 100
    cv2.namedWindow('Left Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Left Image", l_img[l_img.shape[0]//2:l_img.shape[0]//2+nrows,:,:])
    cv2.namedWindow('Right Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Right Image", r_img[l_img.shape[0]//2:l_img.shape[0]//2+nrows,:,:])
    cv2.namedWindow('Left Rectified Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Left Rectified Image", l_img_rect[l_img.shape[0]//2:l_img.shape[0]//2+nrows,:,:])
    cv2.namedWindow('Right Rectified Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Right Rectified Image", r_img_rect[l_img.shape[0]//2:l_img.shape[0]//2+nrows,:,:])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Choose disparity range (num_pixels to look left or right)
    d_min, d_max = (0,50)

    # For all pixels compute best disparity
    print("Computing Disparity...")
    patch_sz = 10
    stp_sz = 1
    r_disp_img = np.zeros((r_img_rect.shape[0],r_img_rect.shape[1]))
    for l_row in range(patch_sz//2, l_img.shape[0]-patch_sz//2+1, stp_sz):
        print(f"Beginning row {l_row} out of {l_img.shape[0]-patch_sz//2+1}")
        for l_col in range(patch_sz//2, l_img.shape[1]-patch_sz//2+1, stp_sz):
            pix_disp = calc_disparity(l_img_rect, r_img_rect, l_row, l_col, patch_sz, d_max)
            r_disp_img[l_row,l_col] = pix_disp
    cv2.normalize(r_disp_img, r_disp_img, 0, 255, cv2.NORM_MINMAX)

    print("OpenCV Disparity...")
    # Set up stereo block matching parameters
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16, blockSize=patch_sz)
    # Compute disparity map
    disparity = stereo.compute(l_img_rect, r_img_rect)
    # Optionally, normalize the disparity map
    cv2.normalize(disparity, disparity, 0, 255, cv2.NORM_MINMAX)
    
    cv2.namedWindow('Left Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Left Image", l_img)
    cv2.namedWindow('Right Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Right Image", r_img)
    cv2.namedWindow('Left Rectified Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Left Rectified Image", l_img_rect)
    cv2.namedWindow('Right Rectified Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Right Rectified Image", r_img_rect)

    cv2.namedWindow('Right Disparity Map', cv2.WINDOW_NORMAL)
    cv2.imshow("Right Disparity Map", r_disp_img)

    cv2.namedWindow('OpenCV Disparity Map', cv2.WINDOW_NORMAL)
    cv2.imshow("OpenCV Disparity Map", disparity)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Remove Outliers
    # print("Removing Outliers...")
    # patch_sz = 100
    # l_disp_img = np.zeros((l_img_rect.shape[0],l_img_rect.shape[1]))
    # for r_row in range(patch_sz//2, r_img.shape[0]-patch_sz//2+1, 1):
    #     print(f"Beginning row {r_row} out of {r_img.shape[0]-patch_sz//2+1}")
    #     for r_col in range(patch_sz//2, r_img.shape[1]-patch_sz//2+1, 1):
    #         pix_disp_ssd, pix_col_ssd = calc_disparity(r_img_rect, l_img_rect, r_row, r_col, patch_sz)
    #         l_disp_img[r_row,pix_col_ssd] = pix_disp_ssd
    # rl_disp = cv2.bitwise_and(r_disp_img, l_disp_img)

    # Obtain depth map from disparity
    print("Extracting depth from disparity map...")
    f = K[0,0]
    b = np.linalg.norm(T)
    depth_map = np.zeros_like(r_disp_img)
    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            depth_map[i,j] = f * b / r_disp_img[i,j]
    
    cv2.namedWindow('Right Depth Map', cv2.WINDOW_NORMAL)
    cv2.imshow("Right Depth Map", depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ## Project image to 3D space:
    # _3d_img = cv2.reprojectImageTo3D(r_disp_img, Q)