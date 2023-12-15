import numpy as np
import cv2

class RobustHomography():
    def __init__(self,n_features,s,eps,N):
        '''
        For RANSAC Algorithm
        s: number of samples
        eps: error threshold (in pixels in the case of images)
        N: number of iterations
        '''
        self.sift = cv2.SIFT_create(n_features)
        self.bfm = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.s = s
        self.eps = eps
        self.N = N
        return
    
    def compute_robust_homography(self,img1,img2,useOpenCV=False):
        # Detect features
        print("Detecting features")
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        print("Found",len(kp1),"features")
        
        # Find matches
        print("Finding feature pairs")
        # good = []
        matches = self.bfm.match(des1, des2)
        # for m, n in matches:
        #     if m.distance < 0.75 * n.distance:
        #         good.append(m)
        # pts1 = [kp1[m.queryIdx].pt for m in good]
        # pts2 = [kp2[m.trainIdx].pt for m in good]
        # print("Found",len(pts1),"feature pairs")
        matches = sorted(matches, key=lambda x: x.distance)
        N_MATCHES = 20
        top_matches = matches[:N_MATCHES]
        if useOpenCV:
            # Extract location of good matches
            points1 = np.zeros((len(top_matches), 2), dtype=np.float32)
            points2 = np.zeros((len(top_matches), 2), dtype=np.float32)

            for i, match in enumerate(top_matches):
                points1[i, :] = kp1[match.queryIdx].pt
                points2[i, :] = kp2[match.trainIdx].pt
            homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
            # print("OpenCV Way")
            return homography
        else:
            pts1 = [kp1[m.queryIdx].pt for m in top_matches]
            pts2 = [kp2[m.trainIdx].pt for m in top_matches]

            # # Run RANSAC to find robust homography
            # print("Running RANSAC for robust homography calculations")
            # _, M = self.run_ransac(pts1,pts2)

            # # Refine RANSAC
            # print("Computing refined homography based on inliers")
            # h = self.homography(M[0], M[1])
            # print("My way")
            h = self.homography(pts1,pts2)
            H = self.reshape_homography(h)
            return H

    def homography(self,pts1,pts2):
        A = np.zeros((2*len(pts1),9))
        for i in range(len(pts1)):
            xs, ys = pts1[i][0], pts1[i][1]
            xd, yd = pts2[i][0], pts2[i][1]
            A[i*2,:] = [xs, ys, 1, 0, 0, 0, -xd*xs, -xd*ys, -xd]
            A[i*2+1,:] = [0, 0, 0, xs, ys, 1, -yd*xs, -yd*ys, -yd]
        AtA = A.T @ A
        eval, evecs = np.linalg.eig(AtA)
        min_eval_idx = np.argmin(eval)
        h = evecs[:,min_eval_idx]
        h /= h[-1]
        # print("Should be 0:\n",A @ h)
        return h
        # U,S,Vt=np.linalg.svd(A)
        # H = Vt[-1,:].reshape(3,3)
        # H = H / H[-1,-1]
    
        # h, res,rank, s = np.linalg.lstsq(AtA - np.eye(9)*eval, np.zeros((9,)), rcond=None)
        # print(h,s)
        
        # ## Dane's way
        # A = np.zeros((2*len(pts1),8))
        # Xp = np.zeros((2*len(pts1),2))
        # for i in range(len(pts1)):
        #     A[i*2,:] = [pts1[i][0], pts1[i][1], 1,
        #                 0,0,0,
        #                 -pts2[i][0]*pts1[i][0], -pts2[i][0]*pts1[i][1]]
        #     A[i*2+1,:] = [0,0,0,
        #                 pts1[i][0], pts1[i][1], 1,
        #                 -pts2[i][1]*pts1[i][0], -pts2[i][1]*pts1[i][1]]
        #     Xp[i*2,:] = [pts2[i][0],pts2[i][1]] 
        # h = np.linalg.lstsq(A, Xp, rcond=None)
        # h = np.array([h[0][:,0]])[0]
        # h = np.concatenate((h,np.array([1.])))
        # return h

    def reshape_homography(self,h):
        H = np.array([
            [h[0],h[1],h[2]],
            [h[3],h[4],h[5]],
            [h[6],h[7],h[8]],
        ])
        return H
    
    def pixelwise_mapping(self,H,img1):
        warped_img1 = np.zeros_like(img1)
        for y in range(img1.shape[0]):
            for x in range(img1.shape[1]):
                pt1 = np.array([x,y,1])
                warp_pt1 = H @ pt1
                warp_pt1 /= warp_pt1[-1]
                new_y, new_x = int(warp_pt1[1]),int(warp_pt1[0])
                if new_y >= 0 and new_x >= 0 and new_y < warped_img1.shape[0] and new_x < warped_img1.shape[1]:
                    warped_img1[int(warp_pt1[1]),int(warp_pt1[0]),:] = img1[y,x,:]
        return warped_img1
    
    def run_ransac(self,data1,data2):
        '''
        RANSAC Algorithm
        Returns:
        :h: 1D homography array
        :M: inliers point pairs
        '''
        h_best = None
        M_best = 0
        d_idxs = np.arange(len(data1),dtype=int)
        for n in range(self.N):
            print("Running iteration",n,"of RANSAC...")
            sampled_d_idxs = np.random.choice(d_idxs,self.s,replace=False)
            ds1 = [data1[idx] for idx in sampled_d_idxs]
            ds2 = [data2[idx] for idx in sampled_d_idxs]
            h = self.homography(ds1,ds2)
            M = self.count_inliers(h,data1,data2)
            print("Num inliears:",M)
            if M > M_best:
                print("Found new best model")
                M_best = M
                h_best = h
        M1, M2 = self.count_inliers(h_best,data1,data2,True)
        return h_best, (M1, M2)

    def count_inliers(self,h,pts1,pts2,get_pairs=False):
        H = self.reshape_homography(h)
        if get_pairs:
            M1 = []
            M2 = []
        M = 0
        for i in range(len(pts1)):
            P1 = np.array([[pts1[i][0], pts1[i][1], 1]]).T
            P2 = np.array([[pts2[i][0], pts2[i][1], 1]]).T
            P2_est = H @ P1
            err = np.linalg.norm(P2 - P2_est)
            # print("Error",err)
            if err < self.eps:
                M += 1
                if get_pairs:
                    M1.append(pts1[i])
                    M2.append(pts2[i])
        if get_pairs:
            return M1, M2
        else:
            return M
        
    def rectify_imgs(self,img1,img2):
        H1to2 = self.compute_robust_homography(img1,img2,True)
        transformed_img1 = cv2.warpPerspective(img1, H1to2, (img1.shape[1], img1.shape[0]))    
        return transformed_img1, img2

if __name__ == '__main__':
    # Load Homography Class
    n_features = 1000
    s = 4
    eps = 100
    N = 100
    rob_hom = RobustHomography(n_features,s,eps,N)

    # Load images
    img1 = cv2.imread("data/simple_objects/house/images/IMG_8988.jpg")
    img2 = cv2.imread("data/simple_objects/house/images/IMG_8989.jpg")

    img1 = cv2.resize(img1,(int(img1.shape[0]*0.1),int(img1.shape[1]*0.1)))
    img2 = cv2.resize(img2,(int(img2.shape[0]*0.1),int(img2.shape[1]*0.1)))

    # Compute Homography
    H = rob_hom.compute_robust_homography(img1,img2,useOpenCV=True)
    print("H:",H)

    # Use the homography matrix to transform the first image to align with the second
    transformed_img1 = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))
    # transformed_img1 = rob_hom.pixelwise_mapping(H,img1)

    cv2.imshow("Image 2: Original",img2)
    cv2.imshow("Image 1: Original",img1)
    cv2.imshow("Image 1: Transformed",transformed_img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert the transformed image to RGB
    # transformed_img_rgb = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB)
