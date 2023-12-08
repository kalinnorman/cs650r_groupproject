import cv2 as cv
import numpy as np

class FeatureHelper():
    def __init__(self):
        self.sift = cv.SIFT_create()
        self.bfm = cv.BFMatcher()
        self.global_matches = dict()
        self.pairwise_matches = dict()
        self.pairwise_matches_masked = dict()
        return
    
    def compute_matches(self, imgs):
        prev_img_name = None
        prev_img = None
        curr_img_name = None
        curr_img = None
        for img_name, img in imgs.items():
            if prev_img_name is None:
                prev_img_name = img_name
                prev_img = img
                continue
            curr_img_name = img_name
            curr_img = img

            # Extract features
            kp_p, des_p = self.sift.detectAndCompute(prev_img, None)
            kp_c, des_c = self.sift.detectAndCompute(curr_img, None)

            # Find matches
            matches = self.bfm.knnMatch(des_p, des_c, k=2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    key = (prev_img_name, curr_img_name)
                    if key in self.pairwise_matches:
                        self.pairwise_matches[(prev_img_name, curr_img_name)].append(
                        (kp_p[m.queryIdx].pt, kp_c[m.trainIdx].pt))
                    else:
                        self.pairwise_matches[(prev_img_name, curr_img_name)] = [
                            (kp_p[m.queryIdx].pt, kp_c[m.trainIdx].pt)
                        ]

            # Update prev variables
            prev_img_name = curr_img_name
            prev_img = curr_img    
        return
    
    def estimate_pairwise_poses_and_3d_points(self, K):
        # Initialize variables
        prv_pts = None
        cur_pts = None
        Rs = []
        ts = []
        pts_3d = []
        # Loop through matches
        for key, value in self.pairwise_matches.items():

            ## Pairwise pose estimation
            
            # Pull out points from matches
            prv_pts = np.array([pt[0] for pt in value])
            cur_pts = np.array([pt[1] for pt in value])
            # Estimate the Essential matrix and recover pose from the Essential matrix
            E, mask = cv.findEssentialMat(prv_pts, cur_pts, cameraMatrix=K, method=cv.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv.recoverPose(E, prv_pts, cur_pts, cameraMatrix=K, mask=mask)
            # Apply mask and get matches that were used to estimate pose (inliers)
            self.pairwise_matches_masked[key] = [value[i] for i in range(len(value)) if mask[i] == 1]
            prv_pts_masked = np.array([pt[0] for pt in self.pairwise_matches_masked[key]])
            cur_pts_masked = np.array([pt[1] for pt in self.pairwise_matches_masked[key]])

            # Rotation and translation are from previous camera to current camera, 
            # but we want to know the rotation and translation from the very first camera to the current camera
            if len(Rs) == 0:
                R_first_cam_to_current_cam = R
                t_first_cam_to_current_cam = t
            else:
                R_first_cam_to_current_came = R @ Rs[-1]
                t_first_cam_to_current_cam = R @ ts[-1] + t
            # Append pose to lists
            Rs.append(R_first_cam_to_current_cam)
            ts.append(t_first_cam_to_current_cam)

            ## Triangulate points

            # Set up projection matrices (setting the first camera as the origin, and using the pose estimates for all others)
            if len(Rs) == 0:
                P1 = K @ np.eye(3, 4)
            else:
                P1 = K @ np.hstack((Rs[-2], ts[-2]))
            P2 = K @ np.hstack((Rs[-1], ts[-1]))
            # Perform triangulation and convert to inhomoegeneous coordinates
            pts_4d = cv.triangulatePoints(P1, P2, prv_pts_masked.T, cur_pts_masked.T)
            pts_3d = pts_4d[:3,:] / pts_4d[3,:]
            # Find any 3d points that match with existing 3d points and average the estimated locations
            # TODO

        return Rs, ts
    


