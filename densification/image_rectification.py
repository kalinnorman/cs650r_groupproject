import cv2

class ImageRectification():
    def __init__(self, cam_mat, dist_coeff):
        self.K = cam_mat
        self.dist = dist_coeff
        self.new_K = None
        return
    
    def rectify(self, left_img, right_img, R, T):
        '''
        Performs Image Rectification on a left and right image pair. 
        Purpose is to put the image planes parallel to each other with parallel epipolar lines. 
        Input:
            left_img: left image of object
            right_img: image where object appears to have shifted left
            R: 3x3 rotation matrix from left camera coordinates to right camera coordinates
            T: translation vector from left camera coordinates to right camera coordinates 
        Output:
            left_img_rectified: a rectified version of the left image
            right_img_rectified: a rectified version of the right image
        '''
        # Asserts same size images
        assert(left_img.shape == right_img.shape)
        w, h = left_img.shape[1], left_img.shape[0]
        # Assumes same camera (i.e. same calibration and distortion coefficients)
        self.new_K, roi = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 1, (w, h))

        rectify_scale= 1
        rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roi_l, roi_r = cv2.stereoRectify(self.new_K, self.dist, self.new_K, self.dist, (w, h), R, T, rectify_scale, (0,0))
        # for elem in Q:
        #     print(np.round(elem))

        stereo_map_l = cv2.initUndistortRectifyMap(self.new_K, self.dist, rect_l, proj_mat_l, (w, h), cv2.CV_16SC2)
        stereo_map_r = cv2.initUndistortRectifyMap(self.new_K, self.dist, rect_r, proj_mat_r, (w, h), cv2.CV_16SC2)

        left_img_rectified = cv2.remap(left_img, stereo_map_l[0], stereo_map_l[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        right_img_rectified = cv2.remap(right_img, stereo_map_r[0], stereo_map_r[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        return left_img_rectified, right_img_rectified
    