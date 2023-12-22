import numpy as np
import matplotlib.pyplot as plt

import time

class BundleAdjustment():
    def __init__(self):
        self.K = None
        self.pts_3d = None
        self.pts_3d_idxs = None
        self.num_3d_pts = None
        self.rot_params = None
        self.num_rot_params = 4
        self.trans_params = None
        self.num_trans_params = 3
        self.num_imgs = None
        self.pts_2d = None

    def run(self, intrinsics, rotations, translations, full_pts_3d, img_pt_and_kp_list):
        # Isolate 3D points that are not None and store, keeping track of the indices used to tie them to the 2D image points
        self.K = intrinsics
        pts_3d_idxs = []
        pts_3d = []
        self.num_3d_pts = 0
        for i in range(len(full_pts_3d)):
            if full_pts_3d[i] is not None:
                pts_3d_idxs.append(i)
                pts_3d.append(full_pts_3d[i])
                self.num_3d_pts += 1
        self.pts_3d = np.array(pts_3d)
        self.pts_3d_idxs = np.array(pts_3d_idxs)

        # self.K = intrinsics
        # self.pts_3d = full_pts_3d
        # self.num_3d_pts, self.num_imgs, _ = img_pt_and_kp_list.shape
        # pts_2d = np.zeros((self.num_3d_pts, self.num_imgs, 3))
        # for i in range(self.num_3d_pts):
        #     for j in range(self.num_imgs):
        #         if not np.isnan(img_pt_and_kp_list[i,j,0]):
        #             pts_2d[i,j,0] = 1
        #             pts_2d[i,j,1:] = img_pt_and_kp_list[i,j,:]
        # self.pts_2d = pts_2d

        # Convert rotations to quaternions and store
        rot_params = []
        for R in rotations:
            rot_params.append(self._convert_rotation_mtx_to_unit_quaternion(R))
        self.rot_params = np.array(rot_params).flatten()
        # Store translations and compute number of images
        trans_params = []
        self.num_imgs = 0
        for t in translations:
            trans_params.append(t.flatten())
            self.num_imgs += 1
        self.trans_params = np.array(trans_params).flatten()
        # Create array mapping 2D image points to 3D points (row i corresponds to 3D point i, column j corresponds to image j)
        pts_2d = np.zeros((self.num_3d_pts, self.num_imgs, 3))
        for j in range(len(img_pt_and_kp_list)):
            temp_list = img_pt_and_kp_list[j]
            for k in range(len(temp_list)):
                idx, pt = temp_list[k]
                i = np.where(self.pts_3d_idxs == idx)[0][0]
                pts_2d[i,j,0] = 1
                pts_2d[i,j,1:] = pt
        self.pts_2d = pts_2d
        # Create vector containing expected measurements (image points)
        x = np.array([])
        for i in range(self.num_3d_pts):
            for j in range(self.num_imgs):
                if self.pts_2d[i,j,0] == 1:
                    x = np.append(x, self.pts_2d[i,j,1:])
        # Create vector containing initial parameters
        p_knot = np.array([])
        for i in range(self.num_imgs):
            p_knot = np.append(p_knot, self.rot_params[self.num_rot_params*i:self.num_rot_params*(i+1)]) # rotation parameters
            p_knot = np.append(p_knot, self.trans_params[self.num_trans_params*i:self.num_trans_params*(i+1)]) # translation parameters
        for i in range(self.num_3d_pts):
            p_knot = np.append(p_knot, self.pts_3d[3*i:3*(i+1)]) # 3D point parameters
        num_proj_params = self.num_rot_params + self.num_trans_params

        jac_cols = num_proj_params * self.num_imgs + 3 * self.num_3d_pts
        ## Setup and run Levenberg-Marquardt
        tau = 10e-3
        eps_1 = 10e-12
        eps_2 = 10e-12
        eps_3 = 10e-12
        eps_4 = 0
        k_max = 100
        k = 0
        v = 2
        p = p_knot.copy()
        # Initial Jacobian and error calc
        t1 = time.time()
        J, xhat = self.calc_J_and_xhat(p, num_proj_params, jac_cols)
        print()
        print("Time to calculate J and xhat:", time.time() - t1)
        eps_p = x - xhat
        A = J.T @ J
        g = J.T @ eps_p
        # print(np.linalg.norm(eps_p))
        # Initialize stop condition
        stop = np.linalg.norm(g, np.inf) < eps_1
        # Initialize mu based on the largest diagonal element of A
        mu = tau * np.max(np.diag(A))
        # LM Loop
        while not stop and k < k_max:
            k += 1
            rho = 0
            while not stop and rho <= 0:
                # Solve for delta_p
                # delta_p = np.linalg.solve(A + mu * np.eye(A.shape[0]), g)
                delta_p = np.linalg.lstsq(A + mu * np.eye(A.shape[0]), g, rcond=None)[0]

                if np.linalg.norm(delta_p) < eps_2 * (np.linalg.norm(p) + eps_2):
                    stop = True
                else:
                    p_new = p + delta_p
                    p_new = self._normalize_quaternions_in_p(p_new, num_proj_params)
                    J, xhat = self.calc_J_and_xhat(p_new, num_proj_params, jac_cols)
                    rho = (np.linalg.norm(eps_p)**2 - np.linalg.norm(x - xhat)**2) / np.dot(delta_p, (mu * delta_p + g))
                    if rho > 0:
                        stop = np.linalg.norm(eps_p) - np.linalg.norm(x - xhat) < eps_4 * np.linalg.norm(eps_p)
                        p = p_new.copy()
                        A = J.T @ J
                        eps_p = x - xhat
                        # print(np.linalg.norm(eps_p))
                        g = J.T @ eps_p
                        stop = stop or np.linalg.norm(g, np.inf) < eps_1
                        mu = mu * max(1/3, 1 - (2 * rho - 1)**3)
                        v = 2
                    else:
                        mu = mu * v
                        v = 2 * v
            stop = np.linalg.norm(eps_p) <= eps_3
        if k >= k_max and not stop:
            print("Bundle Adjustment: Max Iterations Reached in Levengerg-Marquardt")
        # Extract updated rotations, translations and 3D points
        Rs = []
        ts = []
        updated_pts_3d = []
        for i in range(self.num_imgs):
            p_vals = p[num_proj_params*i:num_proj_params*(i+1)]
            Rs.append(self._convert_unit_quaternion_to_rotation_mtx(p_vals[:4]))
            ts.append(p_vals[4:])
        for i in range(self.num_3d_pts):
            updated_pts_3d.append(p[num_proj_params*self.num_imgs + 3*i:num_proj_params*self.num_imgs + 3*(i+1)])
        updated_pts_3d = np.array(updated_pts_3d)
        return Rs, ts, updated_pts_3d

    def project_point(self, K, R, t, pt_3d):
        '''
        Projects a 3D point into 2D space
        Input:
            K (np.array, 3x3) - intrinsic camera calibration matrix
            R (np.array, 3x3) - rotation matrix
            t (np.array, 3,1) - translation vector
            pt_3d (np.array, length 3) - 3D point
        Output:
            pt_2d (np.array, length 2) - 2D point
        '''
        P = K @ np.hstack((R, t))
        pt_3d_homogenous = np.append(pt_3d, 1)
        pt_2d_homogenous = P @ pt_3d_homogenous
        pt_2d = pt_2d_homogenous[:2] / pt_2d_homogenous[2]
        return pt_2d
    
    def project_point_quaternion(self, K, q, t, pt_3d):
        '''
        Projects a 3D point into 2D space
        Input:
            K (np.array, 3x3) - intrinsic camera calibration matrix
            q (np.array, length 4) - unit quaternion of the form [w, x, y, z] -> w + xi + yj + zk
            t (np.array, 3,1) - translation vector
            pt_3d (np.array, length 3) - 3D point
        Output:
            pt_2d (np.array, length 2) - 2D point
        '''
        temp_val = self._quaternion_multiply(self._quaternion_multiply(self._quaternion_inverse(q), np.append(0, pt_3d)), q)
        pt_3d_rotated = temp_val[1:]
        pt_2d_homogenous = K @ (pt_3d_rotated + t)
        pt_2d = pt_2d_homogenous[:2] / pt_2d_homogenous[2]
        return pt_2d

    def calc_J_and_xhat(self, p, num_proj_params, jac_cols):
        # Calculate the Jacobian and xhat
        J = None
        xhat = np.array([])
        for i in range(self.num_3d_pts):
            for j in range(self.num_imgs):
                if self.pts_2d[i,j,0] == 1:
                    # Jacobian
                    J_row = np.zeros((2, jac_cols))
                    subp = p[num_proj_params*j:num_proj_params*(j+1)]
                    subpt3d = p[num_proj_params*self.num_imgs + 3*i:num_proj_params*self.num_imgs + 3*(i+1)]
                    A, B = self._get_jac_row_vals(self.K, subp[:4], subp[4:], subpt3d)
                    J_row[:, 7*j:7*(j+1)] = A
                    J_row[:, 7*self.num_imgs + 3*i:7*self.num_imgs + 3*(i+1)] = B
                    if J is None:
                        J = J_row.copy()
                    else:
                        J = np.vstack((J, J_row))
                    # xhat
                    xhat = np.append(xhat, self.project_point_quaternion(self.K, subp[:4], subp[4:], subpt3d))
        return J, xhat
    
    def _convert_rotation_mtx_to_unit_quaternion(self, R):
        '''
        Converts a rotation matrix to a unit quaternion
        Input:
            R (np.array, 3x3) - rotation matrix
        Output:
            q (np.array, length 4) - unit quaternion of the form [w, x, y, z] -> w + xi + yj + zk
        '''
        q = np.zeros(4)
        t = np.trace(R)
        if t >= 0:
            r = np.sqrt(1 + t)
            s = 1 / (2 * r)
            w = 0.5 * r
            x = (R[2,1] - R[1,2]) * s
            y = (R[0,2] - R[2,0]) * s
            z = (R[1,0] - R[0,1]) * s
            q = np.array([w, x, y, z])
        else:
            if R[1,1] > R[0,0] and R[1,1] > R[2,2]:
                r = np.sqrt(1 + R[1,1] - R[0,0] - R[2,2])
                s = 1 / (2 * r)
                w = (R[2,0] - R[0,2]) * s
                x = 0.5 * r
                y = (R[0,1] + R[1,0]) * s
                z = (R[2,1] + R[1,2]) * s
                q = np.array([w, x, y, z])
            elif R[2,2] > R[0,0]:
                r = np.sqrt(1 + R[2,2] - R[0,0] - R[1,1])
                s = 1 / (2 * r)
                w = (R[0,1] - R[1,0]) * s
                x = (R[0,2] + R[2,0]) * s
                y = 0.5 * r
                z = (R[1,2] + R[2,1]) * s
                q = np.array([w, x, y, z])
            else:
                r = np.sqrt(1 + R[0,0] - R[1,1] - R[2,2])
                s = 1 / (2 * r)
                w = (R[1,2] - R[2,1]) * s
                x = (R[0,2] + R[2,0]) * s
                y = (R[1,0] + R[0,1]) * s
                z = 0.5 * r
                q = np.array([w, x, y, z])
        if np.linalg.norm(q) != 1:
            q = q / np.linalg.norm(q)
        return q
        
    def _convert_unit_quaternion_to_rotation_mtx(self, q):
        '''
        Converts a unit quaternion to a rotation matrix
        Input:
            q (np.array, length 4) - unit quaternion of the form [w, x, y, z] -> w + xi + yj + zk
        Output:
            R (np.array, 3x3) - rotation matrix
        '''
        R = np.zeros((3,3))
        # Verify that q is a unit quaternion
        if np.linalg.norm(q) != 1:
            q = q / np.linalg.norm(q)
        w, x, y, z = q
        R[0,0] = 1 - 2*y**2 - 2*z**2
        R[0,1] = 2*x*y - 2*z*w
        R[0,2] = 2*x*z + 2*y*w
        R[1,0] = 2*x*y + 2*z*w
        R[1,1] = 1 - 2*x**2 - 2*z**2
        R[1,2] = 2*y*z - 2*x*w
        R[2,0] = 2*x*z - 2*y*w
        R[2,1] = 2*y*z + 2*x*w
        R[2,2] = 1 - 2*x**2 - 2*y**2
        return R

    def _quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 - y1*z2 + z1*y2
        y = w1*y2 + x1*z2 + y1*w2 - z1*x2
        z = w1*z2 - x1*y2 + y1*x2 + z1*w2
        q = np.array([w, x, y, z])
        return q
    
    def _quaternion_inverse(self, q):
        w, x, y, z = q
        q_inv = np.array([w, -x, -y, -z])
        return q_inv
    
    def _get_jac_row_vals(self, K, q, t, pt_3d):
        '''
        Calculates the subsection of the jacobian when the projection parameters, and then the 3D point parameters are modified
        Input:
            K (np.array, 3x3) - intrinsic camera calibration matrix
            q (np.array, length 4) - unit quaternion of the form [w, x, y, z] -> w + xi + yj + zk
            t (np.array, length 3) - translation vector
            pt_3d (np.array, length 3) - 3D point
        Output:
            J (np.array, 2x7) - modified projection Jacobian
        '''
        # Calculate A
        A = np.zeros((2,7))
        w, x, y, z = q
        t1, t2, t3 = t
        p = np.array([w, x, y, z, t1, t2, t3])
        delta = 10e-6
        for i in range(7):
            # Central difference approximation
            p_delta = p.copy()
            p_delta[i] = p[i] + delta
            if i < 4:
                p_delta[:4] = p_delta[:4] / np.linalg.norm(p_delta[:4])
            pt_2d_delta_f = self.project_point_quaternion(K, p_delta[:4], p_delta[4:], pt_3d)
            p_delta = p.copy()
            p_delta[i] = p[i] - delta
            if i < 4:
                p_delta[:4] = p_delta[:4] / np.linalg.norm(p_delta[:4])
            pt_2d_delta_b = self.project_point_quaternion(K, p_delta[:4], p_delta[4:], pt_3d)
            A[:,i] = (pt_2d_delta_f - pt_2d_delta_b) / (2 * delta)
        # Calculate B
        B = np.zeros((2,3))
        delta = 10e-6
        for i in range(3):
            # Central difference approximation
            pt_3d_delta = pt_3d.copy()
            pt_3d_delta[i] = pt_3d[i] + delta
            pt_2d_delta_f = self.project_point_quaternion(K, q, t, pt_3d_delta)
            pt_3d_delta[i] = pt_3d[i] - delta
            pt_2d_delta_b = self.project_point_quaternion(K, q, t, pt_3d_delta)
            B[:,i] = (pt_2d_delta_f - pt_2d_delta_b) / (2 * delta)
        return A, B

    def _normalize_quaternions_in_p(self, p, num_params_per_img):
        for i in range(self.num_imgs):
            p_temp = p[num_params_per_img*i:num_params_per_img*(i+1)]
            p_quat = p_temp[:4]
            p_quat = p_quat / np.linalg.norm(p_quat)
            p[num_params_per_img*i:num_params_per_img*i+self.num_rot_params] = p_quat
        return p

    