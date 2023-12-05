import os
import argparse
import json
import pickle as pkl
import cv2
import numpy as np

from image_rectification import ImageRectification
from disparity import Disparity


def resize_img(img1,ratio=1):
    new_width = int(img1.shape[1] * ratio)
    new_height = int(img1.shape[0] * ratio)
    img1 = cv2.resize(img1, (new_width, new_height))
    return img1

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_recon',type=str,help='/path/to/reconstruction.json')
    parser.add_argument('--cam_cal',help='/path/to/camera/calibration/parameters.pkl')
    
    # Arguments
    parser.add_argument('--num_imgs', default=-1, type=int, help='Number of images to perform Depth Estimation on (-1 means all images)')
    parser.add_argument('--img_resize_ratio', default=1, type=float, help='Ratio [0,1] of downsizing the image data') 
    parser.add_argument('--patch_size',default=4,type=int, help="Must be even integer")
    parser.add_argument('--d_max',default=9,type=int)
    return parser

if __name__ == '__main__':
    ## Load Data
    args = vars(read_args().parse_args())
    with open(args['sfm_recon'], 'r') as file:
        reconstruction_data = json.load(file)[0]
    # Order Images by Name
    reconstruction_data["shots"] = {key: reconstruction_data["shots"][key] for key in sorted(reconstruction_data["shots"])}
    cal_file = open(args['cam_cal'],'rb')
    K, dist_coeff = pkl.load(cal_file)
    cal_file.close()

    ## Initialize Depth Classes
    img_rectifier = ImageRectification(K, dist_coeff)
    patch_sz = args['patch_size']
    d_max = args['d_max']
    disp = Disparity(patch_sz, d_max)

    ## Depth Pipeline Begin (left_img=cur_img)
    imgs_filepath = os.path.dirname(args['sfm_recon'])
    imgs_filepath += "/large_images/"#"/shrunk_images/"
    depth_filepath = os.path.dirname(args['sfm_recon'])
    depth_filepath += "/depth_images/"
    os.makedirs(depth_filepath, exist_ok=True)
    

    # stereo = cv2.StereoBM_create(numDisparities=0, blockSize=5)#numDisparites=0, [nD=16, bS=21]
    stereo = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=32,
        blockSize=5,
        uniquenessRatio=5,
        speckleWindowSize=5,
        speckleRange=2, 
        disp12MaxDiff=2,
        P1=8*3*5**2,
        P2=32*3*5**2,
    )


    prev_img = None
    prev_name = None
    prev_R = None
    prev_T = None
    img_ratio = args["img_resize_ratio"]
    img_cnt = 0
    for img_name, img_data in reconstruction_data["shots"].items():
        if img_cnt >= args['num_imgs'] and args['num_imgs'] != -1:
            break

        # Get Image Data
        if prev_img is None:
            print("Starting Image",img_name)
            prev_img = resize_img(cv2.imread(imgs_filepath + img_name), img_ratio) 
            prev_name = img_name
            prev_R, _ = cv2.Rodrigues(np.array(img_data["rotation"]))
            prev_T = np.array(img_data["translation"])
            continue
        
        print("\n----- Depth Estimation for image",img_name,"-----")
        cur_img = resize_img(cv2.imread(imgs_filepath + img_name), img_ratio)
        cur_R, _ = cv2.Rodrigues(np.array(img_data["rotation"]))
        cur_T = np.array(img_data["translation"])

        # Rectify Images
        print("Performing image rectification for image",img_name)
        # R = prev_R @ cur_R.T
        # T = prev_T - cur_T
        R_prev2cur = cur_R @ prev_R.T
        T_prev2cur = cur_T - prev_T
        prev_img_rect, cur_img_rect = img_rectifier.rectify(prev_img, cur_img, R_prev2cur, T_prev2cur)

        l_img = cv2.cvtColor(prev_img,cv2.COLOR_BGR2GRAY)
        r_img = cv2.cvtColor(cur_img,cv2.COLOR_BGR2GRAY)
        depth_map = stereo.compute(l_img,r_img)

        # # # Display Rectified Images
        # # cv2.namedWindow('Current Image', cv2.WINDOW_NORMAL)
        # # cv2.imshow("Current Image", cur_img)
        # # cv2.namedWindow('Previous Image', cv2.WINDOW_NORMAL)
        # # cv2.imshow("Previous Image", prev_img)
        # # cv2.namedWindow('Current Rectified Image', cv2.WINDOW_NORMAL)
        # # cv2.imshow("Current Rectified Image", cur_img_rect)
        # # cv2.namedWindow('Previous Rectified Image', cv2.WINDOW_NORMAL)
        # # cv2.imshow("Previous Rectified Image", prev_img_rect)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()

        # # Compute Image Disparity
        # print("Computing disparity for image",img_name)
        # disparity_img = disp.compute(cur_img, prev_img)

        # # Compute & Save Depth Image
        # print("Computing & Saving dense depth map for image",img_name)
        # f = img_rectifier.new_K[0,0]
        # b = np.linalg.norm(T_prev2cur,2)
        # depth_map = np.zeros_like(disparity_img)
        # for i in range(depth_map.shape[0]):
        #     for j in range(depth_map.shape[1]):
        #         if disparity_img[i,j] > 0:
        #             depth_map[i,j] = f * b / disparity_img[i,j]
        #         else:
        #             depth_map[i,j] = 0
        depth_map = depth_map.astype(np.uint8)
        print(depth_map.dtype)
        colored_image = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
        cv2.imwrite(depth_filepath + "cv_depth_" + img_name[:-4] + "_with_" + prev_name, colored_image)

        # Update previous variables
        prev_img = cur_img
        prev_name = img_name
        prev_R = cur_R
        prev_T = cur_T
        img_cnt += 1
    print("FINISHED DENSE DEPTH MAPS!")