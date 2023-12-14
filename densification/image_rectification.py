import cv2
import numpy as np

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
        T_zeroed = np.zeros_like(T)
        T_zeroed[0:2] = T[0:2]
        # print("T:",T)
        rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roi_l, roi_r = cv2.stereoRectify(self.new_K, self.dist, self.new_K, self.dist, (w, h), R, T_zeroed)#, rectify_scale, (0,0))
        # for elem in Q:
        #     print(np.round(elem))

        stereo_map_l = cv2.initUndistortRectifyMap(self.new_K, self.dist, rect_l, proj_mat_l, (w, h), cv2.CV_16SC2)
        stereo_map_r = cv2.initUndistortRectifyMap(self.new_K, self.dist, rect_r, proj_mat_r, (w, h), cv2.CV_16SC2)

        left_img_rectified = cv2.remap(left_img, stereo_map_l[0], stereo_map_l[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        right_img_rectified = cv2.remap(right_img, stereo_map_r[0], stereo_map_r[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        return left_img_rectified, right_img_rectified
    
    
    def drawlines(self,img1,img2,lines,pts1,pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        rows, c, ch = img1.shape
        # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
        # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
        for r,pt1,pt2 in zip(lines,pts1,pts2):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv2.line(img1, (x0,y0), (x1,y1), color,2)
            img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
            img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
        return img1,img2

    def draw_epipolar_lines(self, img1, img2):
        l_img = np.copy(img1)
        r_img = np.copy(img2)
        
        # Initiate SIFT detector AND Brute-Force Matcher
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher()
        # Extract features
        kp_l, des_l = sift.detectAndCompute(l_img, None)
        kp_r, des_r = sift.detectAndCompute(r_img, None)
        # Find matches
        good = []
        matches = bf.knnMatch(des_l, des_r, k=2)
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        pts_l = np.array([kp_l[m.queryIdx].pt for m in good])
        pts_r = np.array([kp_r[m.trainIdx].pt for m in good])
        pts_l = np.int32(pts_l)
        pts_r = np.int32(pts_r)
        F, mask = cv2.findFundamentalMat(pts_l, pts_r, cv2.FM_RANSAC, confidence=0.9999)
        # Clean up outliers
        # pts_l = pts_l[mask.ravel() == 1]
        # pts_r = pts_r[mask.ravel() == 1]

        ## OpenCV Compute Epipolar Lines
        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(pts_r.reshape(-1,1,2), 2, F)
        lines1 = lines1.reshape(-1,3)
        img5,img6 = self.drawlines(l_img,r_img,lines1,pts_l,pts_r)
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(pts_l.reshape(-1,1,2), 1, F)
        lines2 = lines2.reshape(-1,3)
        img3,img4 = self.drawlines(r_img,l_img,lines2,pts_r,pts_l)
        cv2.imshow("Left Image with Epipolar Lines",img5)
        cv2.imshow("Right Image with Epipolar Lines",img6)
        # cv2.imshow("Left Image with Epipolar Lines",img1)#img5)
        # cv2.imshow("Right Image with Epipolar Lines",img2)#img3)
        # cv2.imwrite("IMG_8990.jpg",img1)
        # cv2.imwrite("IMG_8989.jpg",img2)

        # ## Draw epipolar lines on the left image
        # # Initialize a dictionary to store line colors
        # line_colors = {}

        # # Function to generate a random color
        # def random_color():
        #     return tuple(np.random.randint(0, 255, 3).tolist())
    
        # for i, point_left in enumerate(pts_l):
        #     x, y = point_left
        #     x = int(x)
        #     y = int(y)
        #     # Compute the corresponding epipolar line in the right image
        #     line = np.dot(F, [x, y, 1])
        #     # Calculate two points on the epipolar line
        #     pt1 = (0, int(-line[2] / line[1]))  # (0, y1)
        #     pt2 = (r_img.shape[1], int(-(line[0] * r_img.shape[1] + line[2]) / line[1]))  # (width, y2)
        #     # Generate a random color for the line
        #     color = random_color()
        #     # Store the color in the dictionary
        #     line_colors[i] = color
        #     # Draw the epipolar line on the left image
        #     l_img = cv2.line(l_img, pt1, pt2, line_colors[i], 1)  # Green color, line thickness = 1

        # # Draw epipolar lines on the right image
        # for i, point_right in enumerate(pts_r):
        #     x, y = point_right
        #     x = int(x)
        #     y = int(y)
            
        #     # Compute the corresponding epipolar line in the left image
        #     line = np.dot(F.T, [x, y, 1])
            
        #     # Calculate two points on the epipolar line
        #     pt1 = (0, int(-line[2] / line[1]))  # (0, y1)
        #     pt2 = (l_img.shape[1], int(-(line[0] * l_img.shape[1] + line[2]) / line[1]))  # (width, y2)
            
        #     # Draw the epipolar line on the right image
        #     r_img = cv2.line(r_img, pt1, pt2, line_colors[i], 1)  # Green color, line thickness = 1
        # cv2.imshow("Left Image with Epipolar Lines",l_img)
        # cv2.imshow("Right Image with Epipolar Lines",r_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return