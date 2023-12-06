import argparse
import cv2
import matplotlib.pyplot as plt
import pickle as pkl

def resize_img(img,ratio=1):
    new_width = int(img.shape[1] * ratio)
    new_height = int(img.shape[0] * ratio)
    img = cv2.resize(img, (new_width, new_height))
    return img

def read_args():
    parser = argparse.ArgumentParser()
    
    # Arguments
    parser.add_argument('--cam_cal',type=str,help='/path/to/cam/cal.pkl')
    parser.add_argument('--img_resize_ratio',default=1.0,type=float)
    parser.add_argument('--l_img',type=str,help='/path/to/left/img.jpg')
    parser.add_argument('--r_img',type=str,help='/path/to/right/img.jpg')
    return parser

if __name__ == '__main__':
    ## Load Data
    args = vars(read_args().parse_args())
    l_img = cv2.imread(args['l_img'], cv2.IMREAD_GRAYSCALE)
    r_img = cv2.imread(args['r_img'], cv2.IMREAD_GRAYSCALE)

    ## For custom data images:
    if l_img.shape[0] == 4032 or l_img.shape[0] == 403:
        l_img = cv2.transpose(l_img)
        r_img = cv2.transpose(r_img)
    l_img = resize_img(l_img,args['img_resize_ratio'])
    r_img = resize_img(r_img,args['img_resize_ratio'])
    print("Image Resize Ratio:",args['img_resize_ratio'])
    cal_file = open(args['cam_cal'],'rb')
    K, dist_coeff = pkl.load(cal_file)
    print(K,dist_coeff)
    cal_file.close()
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeff, (l_img.shape[1], l_img.shape[0]), 1, (l_img.shape[1], l_img.shape[0]))
    l_img = cv2.undistort(l_img, K, dist_coeff, None, new_K)
    r_img = cv2.undistort(r_img, K, dist_coeff, None, new_K)

    print("New Image Shapes:",r_img.shape)

    ## Compute Depth Map
    stereo = cv2.StereoBM_create(numDisparities=0, blockSize=21)#numDisparites=0, [nD=16, bS=21]
    # stereo = cv2.StereoSGBM_create(
    #     minDisparity=-1,
    #     numDisparities=32,
    #     blockSize=5,
    #     uniquenessRatio=5,
    #     speckleWindowSize=5,
    #     speckleRange=2, 
    #     disp12MaxDiff=2,
    #     P1=8*3*5**2,
    #     P2=32*3*5**2,
    # )
    depth = stereo.compute(l_img,r_img)
    
    ## Display Results
    cv2.namedWindow('Left', cv2.WINDOW_NORMAL)
    cv2.imshow("Left", l_img)
    cv2.namedWindow('Right', cv2.WINDOW_NORMAL)
    cv2.imshow("Right", r_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.figure()
    plt.imshow(depth)
    plt.axis('off')
    plt.show()


    # cv2.waitKey(0)
    