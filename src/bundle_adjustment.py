import numpy as np

class BundleAdjustment:
    def __init__(self, K, dist) -> None:
        '''
        Initialization for bundle adjustment.

        Parameters
        ----------
        K : np.ndarray
            Intrinsic matrix.
        dist : np.ndarray
            Distortion coefficients.
        '''
        self.K = K
        self.dist = dist

    def run(self, points_2d, camera_poses, points_3d) -> tuple[np.ndarray, np.ndarray]:
        '''
        Runs (hopefully) sparse bundle adjustment on the given data.

        Parameters
        ----------
        points_2d : np.ndarray
            2D points in the image.
        camera_poses : np.ndarray
            Camera poses.
        points_3d : np.ndarray
            3D points in the world.

        Returns:
        --------
        corrected_camera_poses : np.ndarray
            Corrected camera poses.
        corrected_points_3d : np.ndarray
            Corrected 3D points.

        Notes:
        ------
        points_2d should be an array containing all of the 2D image points where each row of the array is of the form: [x, y, camera_id, point_id]
        camera_poses should be an array containing all of the camera poses where each row of the array is of the form: [r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3]
        points_3d should be an array containing all of the 3D points where each row of the array is of the form: [x, y, z] and the index of the row corresponds to the point_id in points_2d
        '''
        # Create initial guess
        
        return None, None

    def _project_3d_point_to_2d(self, point_3d, camera_pose, K, dist_coeffs) -> np.ndarray:
        '''
        Projects a 3D point to 2D.

        Parameters
        ----------
        point_3d : np.ndarray
            3D point.
        camera_pose : np.ndarray
            Camera pose.
        K : np.ndarray
            Intrinsic matrix.
        dist_coeffs : np.ndarray
            Distortion coefficients.

        Returns
        -------
        point_2d : np.ndarray
            2D point.

        Notes:
        ------
        point_3d should be of the form: [x, y, z]
        camera_pose should be of the form: [[r11, r12, r13, t1],
                                            [r21, r22, r23, t2],
                                            [r31, r32, r33, t3]]
        K should be of the form: [[fx,  0, cx],
                                  [ 0, fy, cy],
                                  [ 0,  0,  1]]
        dist_coeffs should be of the form: [k1, k2, p1, p2, k3]
        '''
        # Unpack distortion coefficients
        k1, k2, p1, p2, k3 = dist_coeffs

        # Convert 3D point to homogeneous coordinates
        point_3d_homogeneous = np.append(point_3d, 1)
        # Project 3D point to 2D, noting that normally we would have a projection matrix inbetween K and the camera pose, but we assume that it would be a block matrix of [np.eye(3), np.zeros((3, 1))] which just allows us to treat the camera pose matrix as 3x4 of [R | t] rather than the 4x4 of [R | t; 0 0 0 1]
        point_2d_homogeneous = K @ camera_pose @ point_3d_homogeneous
        # Convert to non-homogeneous coordinates
        point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
        x, y = point_2d
        # Apply distortion
        r = np.sqrt((x - K[0,2])**2 + (y - K[1,2])**2)
        radial_dist = 1 + k1 * r**2 + k2 * r**4 + k3 * r**6
        tangen_dist_x = 2 * p1 * x * y + p2 * (r**2 + 2 * x**2)
        tangen_dist_y = p1 * (r**2 + 2 * y**2) + 2 * p2 * x * y
        x_radial_distortion = x * radial_dist + tangen_dist_x
        y_radial_distortion = y * radial_dist + tangen_dist_y
        # Put back into array
        point_2d_distorted = np.array([x_radial_distortion, y_radial_distortion])
        # Return the projected point
        return point_2d_distorted
    
    def _project_point_vector_input(self, params):
        '''
        Projects a 3D point to 2D.

        Parameters
        ----------
        params : np.ndarray
            Parameters to be passed to _project_3d_point_to_2d.

        Returns
        -------
        point_2d : np.ndarray
            2D point.

        Notes:
        ------
        params should be of the form: [x, y, z, r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3, fx, fy, cx, cy, k1, k2, p1, p2, k3]
        (See function _project_3d_point_to_2d for more details)
        '''
        # Unpack parameters
        point_3d = params[:3]
        camera_pose = params[3:15].reshape((3, 4))
        K_params = params[15:19]
        dist_coeffs = params[19:]
        # Project point
        projected_point_2d = self._project_3d_point_to_2d(point_3d, camera_pose, K_params, dist_coeffs)
        return projected_point_2d
    
    def _reprojection_error(self, params, point_2d):
        '''
        Calculates the reprojection error for a single point.

        Parameters
        ----------
        params : np.ndarray
            Parameters to be passed to _project_point_vector_input.
        point_2d : np.ndarray
            2D point.

        Returns
        -------
        error : float
            Reprojection error.

        Notes:
        ------
        params should be of the form: [x, y, z, r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3, fx, fy, cx, cy, k1, k2, p1, p2, k3]
        (See function _project_3d_point_to_2d for more details)
        point_2d should be of the form: [x, y]
        '''
        # Project point
        projected_point_2d = self._project_point_vector_input(params)
        # Calculate error
        error = np.linalg.norm(point_2d - projected_point_2d)
        return error

if __name__ == "__main__":
    intrinsic_matrix = np.array([[925.79882927,   0.,         635.51907178],
                                 [  0.,         923.71342657, 483.87251378],
                                 [  0.,           0.,           1.        ]])
