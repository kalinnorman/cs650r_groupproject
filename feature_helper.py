import cv2 as cv
import numpy as np
import pickle

class FeatureHelper():
    def __init__(self):
        self.sift = cv.SIFT_create()
        self.bfm = cv.BFMatcher()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        self.flann = cv.FlannBasedMatcher(index_params,search_params)
        # self.flann = cv.FlannBasedMatcher()
        self.global_matches = dict()
        self.pairwise_matches = dict()
        self.pairwise_matches_masked = dict()
        return
    
    def compute_matches(self, imgs):
        '''
        Computes matches between pairs of images, and stores the matches in a dictionary.
        The dictionary is structured as follows:
            - key: tuple of image names
            - value: list of tuples of matched keypoints
        '''
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
    
    def compute_matches_alt(self, imgs):
        '''
        Attempts to create a global list of descriptors, then tracking the indices of the descriptors and keypoints that are found in each image.
        This means that each image can have matches with descriptors from any/all of the previous images, rather than just the one image before it.
        The results are stored in a dictionary:
            - First item
                - key: 'des'
                - value: list of descriptors (built up over time to include all descriptors from all images with (ideally) no duplicates)
            - Second item
                - key: '3d'
                - value: list of 3d points (set to None for now, but will be updated later)
            - Third item and beyond
                - key: image name
                - value: list of three items:
                    - list of indices of all descriptors in the global list of descriptors that are found in the current image
                    - list of two lists:
                        - ordered list of indices of the descriptors in the global list of descriptors that relate to matches found in the current image
                        - ordered list of indices of the keypoints in the current image that correspond with the matches found in the current image
                    - list of all keypoints in the current image
        '''
        for img_name, img in imgs.items():
            # Detect SIFT features
            kp, des = self.sift.detectAndCompute(img, None)
            # If this is the first image, save the descriptors and keypoints
            if len(self.global_matches) == 0:
                self.global_matches['des'] = des
                self.global_matches['3d'] = None
                # Dictionary contains: [all_des_idxs, matches: [[des_idx], [kp_idx]], kp]
                self.global_matches[img_name] = [[i for i in range(len(des))], [[i for i in range(len(des))], [i for i in range(len(kp))]], kp]
                continue
            # On all subsequent images, match the descriptors to the global descriptors
            # matches = self.bfm.knnMatch(queryDescriptors=des, trainDescriptors=self.global_matches['des'], k=2)
            matches = self.flann.knnMatch(queryDescriptors=des, trainDescriptors=self.global_matches['des'], k=2)
            # Filter matches using Lowe's ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            # Determine the indices of the matches
            global_idxs = [m.trainIdx for m in good] # Indices of the global descriptors that match with the current image
            image_idxs = [m.queryIdx for m in good] # Indices of the current image descriptors and keypoints that match with the global descriptors
            # Identify the descriptors not matched in the current image
            unmatched_des_idxs = [i for i in range(len(des)) if i not in image_idxs]
            # Get a subset of the unmatched descriptors and add it to the end of the full list of descriptors
            idx_for_next_des = len(self.global_matches['des'])
            self.global_matches['des'] = np.vstack((self.global_matches['des'], des[unmatched_des_idxs]))
            # Identify the indices of the global list of the descriptors that correspond with the local list of keypoints
            all_des_idxs = []
            for i in range(len(kp)):
                if i not in image_idxs:
                    all_des_idxs.append(idx_for_next_des)
                    idx_for_next_des += 1
                else:
                    all_des_idxs.append(global_idxs[image_idxs.index(i)])
            # Add to the dictionary
            self.global_matches[img_name] = [all_des_idxs, [global_idxs, image_idxs], kp]
        # Update the 3d to be a list of None's the same length as the descriptors
        self.global_matches['3d'] = [None for i in range(len(self.global_matches['des']))]
    
    def estimate_pairwise_poses_and_3d_points(self, K):
        '''
        Estimates the pairwise poses and 3d points from the matches.
        This is done via estimating the essential matrix, recovering the pose from the essential matrix, and triangulating the points.
        From this we get:
            - Rs: list of rotation matrices, where each rotation matrix is from the global frame to the current camera frame
            - ts: list of translation vectors, where each translation vector is from the global frame to the current camera frame
            - pts_3ds: list of 3d points that is filled into the global_matches dictionary and any 3d points with multiple estimates are averaged
        The rotation list, translation list, and global_matches dictionary are returned.
        '''
        # Initialize variables
        prev_key = None
        Rs = []
        ts = []
        pts_3d = []
        # Loop through matches
        for key, value in self.global_matches.items():
            # Skip the first key (descriptors)
            if key == 'des':
                continue
            # Skip the second key (3d points)
            if key == '3d':
                continue
            # Skip the first image (no previous image to compare to)
            if prev_key is None:
                prev_key = key
                continue
            # Identify the the matches between the current image and previous image
            global_des_idxs_for_cur_img_matches = self.global_matches[key][1][0]
            prev_img_des_idxs = self.global_matches[prev_key][0]
            matching_idxs_cur_to_prev = []
            for i in range(len(global_des_idxs_for_cur_img_matches)):
                if global_des_idxs_for_cur_img_matches[i] in prev_img_des_idxs:
                    matching_idxs_cur_to_prev.append(global_des_idxs_for_cur_img_matches[i]) # Index of descriptor in global list that both images share
            # From the matches, find the correct keypoint indices, then pull out those keypoints
            temp = [self.global_matches[prev_key][1][0].index(i) for i in matching_idxs_cur_to_prev] # Index of where the descriptor index is in the list
            prev_img_kp_idxs = [self.global_matches[prev_key][1][1][i] for i in temp] # Index of the keypoint that corresponds to the same descriptor
            prev_img_kp = [self.global_matches[prev_key][2][i].pt for i in prev_img_kp_idxs] # Keypoint coordinates
            prev_img_kp = np.array(prev_img_kp)
            temp = [self.global_matches[key][1][0].index(i) for i in matching_idxs_cur_to_prev]
            cur_img_kp_idxs = [self.global_matches[key][1][1][i] for i in temp]
            cur_img_kp = [self.global_matches[key][2][i].pt for i in cur_img_kp_idxs]
            cur_img_kp = np.array(cur_img_kp)
            # Estimate the essential matrix and recover pose from the essential matrix
            E, mask = cv.findEssentialMat(prev_img_kp, cur_img_kp, cameraMatrix=K, method=cv.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv.recoverPose(E, prev_img_kp, cur_img_kp, cameraMatrix=K, mask=mask)
            # Get masked points
            prv_pts_masked = prev_img_kp[mask.ravel()==1]
            cur_pts_masked = cur_img_kp[mask.ravel()==1]
            # Rotation and translation are from previous camera to current camera, but we want rotation and translation from first camera to the current camera
            if len(Rs) == 0:
                Rs.append(np.eye(3)) # Add in identity matrix for first camera
                ts.append(np.zeros((3,1))) # Add in zero vector for first camera
                R_first_cam_to_current_cam = R
                t_first_cam_to_current_cam = t
            else:
                R_first_cam_to_current_cam = R @ Rs[-1]
                t_first_cam_to_current_cam = R @ ts[-1] + t
            # Append pose to lists
            Rs.append(R_first_cam_to_current_cam)
            ts.append(t_first_cam_to_current_cam)
            # Set up projection matrices (setting the first camera as the origin, and using the pose estimates for all others)
            P1 = K @ np.hstack((Rs[-2], ts[-2]))
            P2 = K @ np.hstack((Rs[-1], ts[-1]))
            # Triangulate points and convert to inhomoegeneous coordinates
            pts_4d = cv.triangulatePoints(P1, P2, prv_pts_masked.T, cur_pts_masked.T)
            pts_3d = pts_4d[:3,:] / pts_4d[3,:]
            # Now we need to identify the descriptors that match with the 3d points we got from triangulation (taking into account the mask)
            mask_bool_list = mask.ravel() == 1
            kp_idxs_masked = [cur_img_kp_idxs[i] for i in range(len(mask_bool_list)) if mask_bool_list[i] == True]
            des_idxs_masked = [self.global_matches[key][1][0][self.global_matches[key][1][1].index(i)] for i in kp_idxs_masked]
            # Update the 3d points, and if any existing 3d points match with the new 3d points, average the estimated locations
            for i in range(len(des_idxs_masked)):
                if self.global_matches['3d'][des_idxs_masked[i]] is None:
                    self.global_matches['3d'][des_idxs_masked[i]] = pts_3d[:,i]
                else:
                    self.global_matches['3d'][des_idxs_masked[i]] = (self.global_matches['3d'][des_idxs_masked[i]] + pts_3d[:,i]) / 2
        # Return the rotation and translation lists, and the global_matches dictionary
        return Rs, ts, self.global_matches
    


