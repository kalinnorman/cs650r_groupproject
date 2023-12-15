import os
import argparse
import json
import pickle as pkl
import cv2
import numpy as np

import open3d as o3d

from image_rectification import ImageRectification
from disparity import Disparity


def resize_img(img,ratio=1):
    new_width = int(img.shape[1] * ratio)
    new_height = int(img.shape[0] * ratio)
    img = cv2.resize(img, (new_width, new_height))
    return img

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1_filepath',type=str,help='path/to/img1.jpg')
    parser.add_argument('--img2_filepath',type=str,help='path/to/img2.jpg')
    # parser.add_argument('--sfm_recon',type=str,help='/path/to/reconstruction.json')
    # parser.add_argument('--cam_cal',help='/path/to/camera/calibration/parameters.pkl')
    
    # Arguments
    # parser.add_argument('--num_imgs', default=-1, type=int, help='Number of images to perform Depth Estimation on (-1 means all images)')
    # parser.add_argument('--img_resize_ratio', default=1, type=float, help='Ratio [0,1] of downsizing the image data') 
    parser.add_argument('--patch_size',default=21,type=int, help="Must be even integer")
    parser.add_argument('--d_max',default=16*3,type=int)
    return parser

if __name__ == '__main__':
    ## Load Data
    args = vars(read_args().parse_args())
    # with open(args['sfm_recon'], 'r') as file:
    #     reconstruction_data = json.load(file)[0] # grab first reconstruction group (should be all images if dataset is good)
    # # Order Images by Name
    # reconstruction_data["shots"] = {key: reconstruction_data["shots"][key] for key in sorted(reconstruction_data["shots"])}
    # cal_file = open(args['cam_cal'],'rb')
    # K, dist_coeff = pkl.load(cal_file)
    # cal_file.close()
    K = np.eye(3)
    dist_coeff = np.zeros((5,))

    ## Initialize Depth Classes
    img_rectifier = ImageRectification(K, dist_coeff)
    patch_sz = args['patch_size']
    d_max = args['d_max']
    disp = Disparity(patch_sz, d_max)
    stereo = cv2.StereoBM_create(numDisparities=d_max, blockSize=patch_sz)

    ## Load Images
    # img1 = resize_img(cv2.imread(args['img1_filepath']),0.1)
    # img2 = resize_img(cv2.imread(args['img2_filepath']),0.1)
    img1 = cv2.imread(args['img1_filepath'])
    img2 = cv2.imread(args['img2_filepath'])
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    ## Draw Epipolar Geometry
    img_rectifier.draw_epipolar_lines(img1,img2)

    ## Compute Disparity - patch_size=21, d_max=64 for test statue image
    disparity = disp.compute(img1_gray,img2_gray)
    # disparity = disp.compute_disparity_cgpt(img1_gray,img2_gray)
    # disparity = stereo.compute(img1_gray,img2_gray)

    disparity = disparity
    disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity = disparity.astype(np.uint8)
    colored_image = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
    cv2.imshow("Disparity Map",colored_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ## Compute depth
    # print("Computing & Saving dense depth map for image",img_name)
    # f = K[0,0]#img_rectifier.new_K[0,0]
    # b = np.linalg.norm(T_prev2cur,2)
    depth_map = np.zeros_like(disparity)
    colors = []
    points = []
    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            if disparity[i,j] > 0:
                depth_map[i,j] = disparity[i,j]#f * b / disparity[i,j]
            else:
                depth_map[i,j] = 0
            points.append([i,j,depth_map[i,j]])
            colors.append([img1[i,j,2],img1[i,j,1],img1[i,j,0]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors)/255.0)
    o3d.io.write_point_cloud("test1_pc.ply",pcd,write_ascii=True)