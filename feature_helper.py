import cv2 as cv
import numpy as np
import pickle
from tqdm import tqdm

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
            
            # If there are multiple matches to the same global descriptor, remove all of them
            remove_dupes = True
            if remove_dupes:
                for i in range(len(global_idxs)):
                    # Find if the index is repeated
                    if global_idxs.count(global_idxs[i]) > 1:
                        # Identify all indices with that value
                        idxs = [j for j in range(len(global_idxs)) if global_idxs[j] == global_idxs[i]]
                        # Set all of those indices to -1
                        for j in idxs:
                            global_idxs[j] = -1
                            image_idxs[j] = -1
                global_idxs = [i for i in global_idxs if i != -1]
                image_idxs = [i for i in image_idxs if i != -1]
            
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
    
    def estimate_pairwise_poses_and_3d_points(self, K, max_matches_per_img = None):
        '''
        Estimates the pairwise poses and 3d points from the matches.
        This is done via estimating the essential matrix, recovering the pose from the essential matrix, and triangulating the points.
        From this we get:
            - Rs: list of rotation matrices, where each rotation matrix is from the global frame to the current camera frame
            - ts: list of translation vectors, where each translation vector is from the global frame to the current camera frame
            - pts_3ds: list of 3d points that is filled into the global_matches dictionary and any 3d points with multiple estimates are averaged
        The rotation list, translation list, 3d points, and the list of lists of 3d point indices and keypoints per image are returned.
        '''
        # Initialize variables
        prev_key = None
        Rs = []
        ts = []
        pts_3d = []
        img_desidx_kpval_list = []
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
            prev_img_kp_idxs = [self.global_matches[prev_key][0].index(i) for i in matching_idxs_cur_to_prev] # Index in the descriptor list for the previous image, which is also the keypoint index
            prev_img_kp = [self.global_matches[prev_key][2][i] for i in prev_img_kp_idxs] # Keypoint coordinates
            prev_img_kp_pts = [kp.pt for kp in prev_img_kp]
            prev_img_kp_pts = np.array(prev_img_kp_pts)
            temp = [self.global_matches[key][1][0].index(i) for i in matching_idxs_cur_to_prev] # Index in the descriptor list for the current image, which will have a corresponding keypoint index
            cur_img_kp_idxs = [self.global_matches[key][1][1][i] for i in temp] # Corresponding keypoint indices
            cur_img_kp = [self.global_matches[key][2][i] for i in cur_img_kp_idxs] # Keypoint coordinates
            cur_img_kp_pts = [kp.pt for kp in cur_img_kp]
            cur_img_kp_pts = np.array(cur_img_kp_pts)
            # Estimate the essential matrix and recover pose from the essential matrix
            E, mask = cv.findEssentialMat(prev_img_kp_pts, cur_img_kp_pts, cameraMatrix=K, method=cv.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv.recoverPose(E, prev_img_kp_pts, cur_img_kp_pts, cameraMatrix=K, mask=mask)
            # Get masked points
            prv_pts_masked = prev_img_kp_pts[mask.ravel()==1]
            cur_pts_masked = cur_img_kp_pts[mask.ravel()==1]
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
            cur_img_kp_masked = [cur_img_kp[i] for i in range(len(mask_bool_list)) if mask_bool_list[i] == True] 
            kp_idxs_masked = []
            des_idxs_masked = []
            for mkp in cur_img_kp_masked:
                kpidx = self.global_matches[key][2].index(mkp)
                kp_idxs_masked.append(kpidx)
                des_idxs_masked.append(self.global_matches[key][1][0][self.global_matches[key][1][1].index(kpidx)])

            # Limit the number of matches to a maximum value
            if max_matches_per_img is not None and len(des_idxs_masked) > max_matches_per_img:
                temp_idxs_list = list(range(len(des_idxs_masked)))
                temp_idxs_to_keep = np.random.choice(temp_idxs_list, size=max_matches_per_img, replace=False)
                des_idxs_masked = [des_idxs_masked[i] for i in temp_idxs_to_keep]
                kp_idxs_masked = [kp_idxs_masked[i] for i in temp_idxs_to_keep]
                pts_3d = pts_3d[:,temp_idxs_to_keep]

            # Update the 3d points, and if any existing 3d points match with the new 3d points, average the estimated locations
            for i in range(len(des_idxs_masked)):
                if self.global_matches['3d'][des_idxs_masked[i]] is None:
                    self.global_matches['3d'][des_idxs_masked[i]] = pts_3d[:,i]
                else:
                    self.global_matches['3d'][des_idxs_masked[i]] = (self.global_matches['3d'][des_idxs_masked[i]] + pts_3d[:,i]) / 2
            # Add info for the latest image pair to the list
            prev_img_list = []
            for i in range(len(des_idxs_masked)):
                if [des_idxs_masked[i], prv_pts_masked[i]] not in prev_img_list:
                    prev_img_list.append([des_idxs_masked[i], prv_pts_masked[i]])
            cur_img_list = []
            for i in range(len(des_idxs_masked)):
                if [des_idxs_masked[i], cur_pts_masked[i]] not in cur_img_list:
                    cur_img_list.append([des_idxs_masked[i], cur_pts_masked[i]])
            if len(img_desidx_kpval_list) == 0:
                img_desidx_kpval_list.append(prev_img_list)
            else:
                for templist in prev_img_list:
                    try:
                        if templist not in img_desidx_kpval_list[-1]:
                            img_desidx_kpval_list[-1].append(templist)
                    except Exception as e: # If the descriptor matches, but the keypoint doesn't, it throws and error and we want to remove those from the list to avoid bad matches (if possible)
                        if 'truth value of an array with more than one element' in str(e):
                            # If the descriptor matches, but the keypoint doesn't, remove the descriptor from both lists
                            img_desidx_kpval_list[-1] = [i for i in img_desidx_kpval_list[-1] if i[0] != templist[0]]
                            cur_img_list = [i for i in cur_img_list if i[0] != templist[0]]
                        else:
                            raise e
            img_desidx_kpval_list.append(cur_img_list)
            # Update the previous key
            prev_key = key
        # Return the rotation and translation lists, 3d points, and the list of lists of descriptors and keypoints
        return Rs, ts, self.global_matches['3d'], img_desidx_kpval_list
    
    def est_stuff(self, imgs, K):
        # Initialization
        img_names = list(imgs.keys())
        kp, des = [], []
        Rs, ts = [], []
        pts_3d = None
        pts_3d_to_kpts = None
        # Identify all SIFT features for each image
        for i in tqdm(range(len(img_names))):
            kptemp, destemp = self.sift.detectAndCompute(imgs[img_names[i]], None)
            kp.append(kptemp)
            des.append(destemp)
        # Exhaustive matching between all images, identifying everything up through triangulated points
        for i in tqdm(range(len(img_names)-1)):
            # Rtemp, ttemp = [], []
            kp1, des1 = kp[i], des[i]
            for j in range(i+1, len(img_names)):
                kp2, des2 = kp[j], des[j]
                # Find all matches
                matches = self.bfm.knnMatch(queryDescriptors=des1, trainDescriptors=des2, k=2)
                queryidxs = []
                trainidxs = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        queryidxs.append(m.queryIdx)
                        trainidxs.append(m.trainIdx)
                # Estimate the essential matrix and recover relative pose from the essential matrix
                matched_kp1 = np.array([kp1[i].pt for i in queryidxs])
                matched_kp2 = np.array([kp2[i].pt for i in trainidxs])
                E, mask = cv.findEssentialMat(matched_kp1, matched_kp2, cameraMatrix=K, method=cv.RANSAC, prob=0.999, threshold=1.0)
                _, R, t, mask = cv.recoverPose(E, matched_kp1, matched_kp2, cameraMatrix=K, mask=mask)
                # Treat img 1 as the origin, and convert R and t to be a global represenation
                if i == 0: # On the first iteration we are doing everything compared to img 1, the origin
                    if j == i+1:
                        Rs.append(np.eye(3))
                        ts.append(np.zeros((3,1)))
                    Rs.append(R)
                    ts.append(t)
                else: # On all other iterations we are comparing to the prev image and need to convert to global
                    R1toj = R @ Rs[i]
                    Rs[j] = R1toj # Update previous global estimate with new compound one (more matches so we trust newer estimate more)
                # Triangulate 3d points using masked 2d points
                P2 = K @ np.hstack((Rs[j], ts[j]))
                P1 = K @ np.hstack((Rs[i], ts[i]))
                pts1_masked = matched_kp1[mask.ravel()==1]
                pts2_masked = matched_kp2[mask.ravel()==1]
                pts4d = cv.triangulatePoints(P1, P2, pts1_masked.T, pts2_masked.T)
                pts3d = pts4d[:3,:] / pts4d[3,:]
                pts3d = pts3d.T
                if pts_3d is None:
                    pts_3d = pts3d
                    pts_3d_to_kpts = np.empty((len(pts3d), len(img_names), 2))
                    pts_3d_to_kpts.fill(np.nan)
                    for k in range(len(pts3d)):
                        pts_3d_to_kpts[k,i,:] = pts1_masked[k]
                        pts_3d_to_kpts[k,j,:] = pts2_masked[k]
                else:
                    # Find any existing 3d points that match with the triangulated points
                    existing_3d_pt_idxs = []
                    corresponding_2d_idxs = []
                    for k, pt in enumerate(pts1_masked):
                        for l, val in enumerate(pts_3d_to_kpts[:,i,:]):
                            if np.all(np.equal(val, pt)):
                                existing_3d_pt_idxs.append(l)
                                corresponding_2d_idxs.append(k)
                    # Update existing items with the new triangulated points
                    for k, l in zip(corresponding_2d_idxs, existing_3d_pt_idxs):
                        # Average the triangulated point with the existing point
                        pts_3d[l] = (pts_3d[l] + pts3d[k]) / 2
                        # Update the 2d point correspondences
                        pts_3d_to_kpts[l,j,:] = pts2_masked[k]
                    # Remove the items we've already handled from the new triangulated points and 2d points
                    pts3d = np.delete(pts3d, corresponding_2d_idxs, axis=0)
                    pts1_masked = np.delete(pts1_masked, corresponding_2d_idxs, axis=0)
                    pts2_masked = np.delete(pts2_masked, corresponding_2d_idxs, axis=0)
                    # Do it again for the other image
                    existing_3d_pt_idxs = []
                    corresponding_2d_idxs = []
                    for k, pt in enumerate(pts2_masked):
                        for l, val in enumerate(pts_3d_to_kpts[:,j,:]):
                            if np.all(np.equal(val, pt)):
                                existing_3d_pt_idxs.append(l)
                                corresponding_2d_idxs.append(k)
                    # Update existing items with the new triangulated points
                    for k, l in zip(corresponding_2d_idxs, existing_3d_pt_idxs):
                        # Average the triangulated point with the existing point
                        pts_3d[l] = (pts_3d[l] + pts3d[k]) / 2
                        # Update the 2d point correspondences
                        pts_3d_to_kpts[l,i,:] = pts1_masked[k]
                    # Remove the items we've already handled from the new triangulated points and 2d points
                    pts3d = np.delete(pts3d, corresponding_2d_idxs, axis=0)
                    pts1_masked = np.delete(pts1_masked, corresponding_2d_idxs, axis=0)
                    pts2_masked = np.delete(pts2_masked, corresponding_2d_idxs, axis=0)
                    # Add the new points to the two arrays
                    temp_3dtokp = np.empty((len(pts3d), len(img_names), 2))
                    temp_3dtokp.fill(np.nan)
                    temp_3dtokp[:,i,:] = pts1_masked
                    temp_3dtokp[:,j,:] = pts2_masked
                    pts_3d = np.vstack((pts_3d, pts3d))
                    pts_3d_to_kpts = np.vstack((pts_3d_to_kpts, temp_3dtokp))
        # Trim the results to only keep 1000 points
        pts_3d_trimmed = np.zeros((1000,3))
        pts_3d_to_kpts_trimmed = np.empty((1000, len(img_names), 2))
        pts_3d_to_kpts_trimmed.fill(np.nan)
        idxs_used = []
        idxs_to_choose_from = list(range(len(pts_3d)))
        cur_idx = 0
        num_pts_shared = len(img_names)
        num_imgs = len(img_names)
        while cur_idx < 1000:
            rand_idxs = np.random.choice(idxs_to_choose_from, len(idxs_to_choose_from), replace=False)
            for rand_idx in rand_idxs:
                if rand_idx not in idxs_used:
                    if np.sum(np.isnan(pts_3d_to_kpts[rand_idx,:,0])) <= num_imgs - num_pts_shared:
                        idxs_used.append(rand_idx)
                        pts_3d_trimmed[cur_idx,:] = pts_3d[rand_idx,:]
                        pts_3d_to_kpts_trimmed[cur_idx,:,:] = pts_3d_to_kpts[rand_idx,:,:]
                        cur_idx += 1
                        if cur_idx >= 1000:
                            break
            num_pts_shared -= 1

        pts_3d = pts_3d_trimmed
        pts_3d_to_kpts = pts_3d_to_kpts_trimmed

        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # # Create 2d image to show where the points are
        # ro, co, _ = pts_3d_to_kpts.shape
        # pts_3d_to_kpts_img = np.zeros((ro, co))
        # # Assign a value of 1 if there is a keypoint at that location
        # for i in range(len(pts_3d_to_kpts_img)):
        #     for j in range(len(pts_3d_to_kpts_img[i])):
        #         if not np.isnan(pts_3d_to_kpts[i,j,0]):
        #             pts_3d_to_kpts_img[i,j] = 1
        # # Show the image
        # plt.imshow(pts_3d_to_kpts_img)
        # plt.show()

        # Return the results
        return Rs, ts, pts_3d, pts_3d_to_kpts