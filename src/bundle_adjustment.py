import numpy as np
from enum import Enum

class ProjectionMethod(Enum):
    '''
    Enumeration of the different methods of projection.
    '''
    PROJECTION_MATRIX = 0 # Solve the 12 parameters of the projection matrix, which is created by K @ [R | t], and the 3 parameters of the 3D point
    PROJECTION_MATRIX_WITH_DISTORTION = 1 # Solve for the 12 paramters of the projection matrix, the 5 distortion parameters, and the 3 parameters of the 3D point
    DECOMPOSED_PROJECTION_MATRIX = 2 # Solve for the 4 camera intrinsics parameters, the 9 camera rotation parameters, the 3 camera translation parameters, and the 3 parameters of the 3D point
    DECOMPOSED_PROJECTION_MATRIX_WITH_DISTORTION = # Solve for the 4 camera intrinsics parameters, the 9 camera rotation parameters, the 3 camera translation parameters, the 5 distortion parameters, and the 3 parameters of the 3D point

class BundleAdjustment:
    def __init__(self, K, dist) -> None:
        pass

    def run(self, points_2d, camera_poses, points_3d, method, K, distortion_params=None) -> tuple[np.ndarray, np.ndarray]:
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
        method : ProjectionMethod
            Projection method to use.
        K : np.ndarray
            Intrinsic camera parameters (3x3 array).
        distortion_params : np.ndarray, optional
            Distortion parameters (vector of 5 values or 'None' to disable).

        Returns:
        --------

        corrected_camera_poses : np.ndarray
            Corrected camera poses.
        corrected_points_3d : np.ndarray
            Corrected 3D points.

        Notes:
        ------
        points_2d should be an array containing all of the 2D image points where each row of the array is of the form: [x, y, camera_id, point_id]
        camera_poses should be an array containing all of the camera poses where each row of the array is of the form: [r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3] and the index of the row coresponds to the camera_id in points_2d
        points_3d should be an array containing all of the 3D points where each row of the array is of the form: [x, y, z] and the index of the row corresponds to the point_id in points_2d
        method should be a ProjectionMethod enum value, with those options being: PROJECTION_MATRIX, PROJECTION_MATRIX_WITH_DISTORTION, DECOMPOSED_PROJECTION_MATRIX, DECOMPOSED_PROJECTION_MATRIX_WITH_DISTORTION
        '''
        # If the method is meant to use distortion, check that distortion parameters have been passed in
        if method == ProjectionMethod.PROJECTION_MATRIX_WITH_DISTORTION or method == ProjectionMethod.DECOMPOSED_PROJECTION_MATRIX_WITH_DISTORTION:
            if distortion_params is None:
                raise ValueError("Distortion parameters must be provided for this projection method")
        # If the method is NOT a decomposed projection method, combine the camera intrinsics and extrinsics into a single projection matrix
        if method == ProjectionMethod.PROJECTION_MATRIX or method == ProjectionMethod.PROJECTION_MATRIX_WITH_DISTORTION:
            # Create a list with the parameters for the projection matrix for each camera pose
            projection_matrix_params = []
            for camera_pose in camera_poses:
                proj_mtx = K @ camera_pose.reshape((3, 4))
                projection_matrix_params.append(proj_mtx.flatten())
            # Convert the list to an array
            projection_matrix_params = np.array(projection_matrix_params)
        # If the method is a decomposed projection method, convert the camera intrinsic matrix to a vector of the form [fx, fy, cx, cy]
        elif method == ProjectionMethod.DECOMPOSED_PROJECTION_MATRIX or method == ProjectionMethod.DECOMPOSED_PROJECTION_MATRIX_WITH_DISTORTION:
            # Convert the camera intrinsic matrix to a vector
            K = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])


        
        return None, None
    
    def _apply_distortion(self, point_2d, dist_coeffs, K) -> np.ndarray:
        '''
        Applies distortion to a 2D point.

        Parameters
        ----------
        point_2d : np.ndarray
            Inhomogenous 2D image point (vector of 2 values).
        dist_coeffs : np.ndarray
            Distortion coefficients (vector of 5 values).
        K : np.ndarray
            Intrinsic camera parameters (3x3 array).
        
        Returns
        -------
        point_2d_distorted : np.ndarray
            Inhomogenous 2D image point (vector of 2 values).
        '''
        # Unpack distortion coefficients
        k1, k2, p1, p2, k3 = dist_coeffs
        # Apply distortion
        x, y = point_2d
        r = np.sqrt((x - K[0,2])**2 + (y - K[1,2])**2)
        radial_dist = 1 + k1 * r**2 + k2 * r**4 + k3 * r**6
        tangen_dist_x = 2 * p1 * x * y + p2 * (r**2 + 2 * x**2)
        tangen_dist_y = p1 * (r**2 + 2 * y**2) + 2 * p2 * x * y
        x_radial_distortion = x * radial_dist + tangen_dist_x
        y_radial_distortion = y * radial_dist + tangen_dist_y
        # Put back into array
        point_2d_distorted = np.array([x_radial_distortion, y_radial_distortion])
        # Return the distorted point
        return point_2d_distorted
    
    def _project_3d_point_to_2d_via_projection(self, point_3d, projection_matrix, dist_coeffs=None) -> np.ndarray:
        '''
        Projects a 3D point to 2D via pt_2d = P * pt_3d.
        Optionally applies distortion.

        Parameters
        ----------
        point_3d : np.ndarray
            Inhomogenous 3D world point (vector of 3 values).
        projection_matrix : np.ndarray
            Projection matrix (3x4 array).
        dist_coeffs : np.ndarray, optional
            Distortion coefficients (vector of 5 values or 'None' to disable).
            By default this is set to 'None'.
        
        Returns
        -------
        point_2d : np.ndarray
            Inhomogenous 2D image point (vector of 2 values).

        Notes:
        ------
        point_3d should be of the form: [x, y, z]
        projection_matrix should be of the form: [[p11, p12, p13, p14],
                                                  [p21, p22, p23, p24],
                                                  [p31, p32, p33, p34]]
        dist_coeffs should be of the form: [k1, k2, p1, p2, k3], if it is not set to 'None'
        '''

        # Project 3D point to 2D
        point_2d_homogeneous = projection_matrix @ np.append(point_3d, 1)
        # Convert to non-homogeneous coordinates
        point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
        # Apply distortion
        if dist_coeffs is not None:
            point_2d = self._apply_distortion(point_2d, dist_coeffs)
        # Return the projected point
        return point_2d


    def _project_3d_point_to_2d_via_decomposed_projection(self, point_3d, K, camera_pose, dist_coeffs=None) -> np.ndarray:
        '''
        Projects a 3D point to 2D via pt_2d = K * [R | t] * pt_3d.
        Optionally applies distortion.

        Parameters
        ----------
        point_3d : np.ndarray
            Inhomogenous 3D world point (vector of 3 values).
        K : np.ndarray
            Intrinsic camera parameters (3x3 array).
        camera_pose : np.ndarray
            Camera pose (3x4 array of the form [R | t] where R is a 3x3 rotation matrix and t is a 3x1 translation vector).
        dist_coeffs : np.ndarray, optional
            Distortion coefficients (vector of 5 values or 'None').
            By default this is set to 'None'.

        Notes:
        ------
        point_3d should be of the form: [x, y, z]
        K should be of the form: [[fx,  0, cx],
                                  [ 0, fy, cy],
                                  [ 0,  0,  1]]
        camera_pose should be of the form: [[r11, r12, r13, t1],
                                            [r21, r22, r23, t2],
                                            [r31, r32, r33, t3]]
        dist_coeffs should be of the form: [k1, k2, p1, p2, k3], if it is not set to 'None'
        '''
        # Convert 3D point to homogeneous coordinates
        point_3d_homogeneous = np.append(point_3d, 1)
        # Project 3D point to 2D, noting that normally we would have a projection matrix inbetween K and the camera pose, but we assume that it would be a block matrix of [np.eye(3), np.zeros((3, 1))] which just allows us to treat the camera pose matrix as 3x4 of [R | t] rather than the 4x4 of [R | t; 0 0 0 1]
        point_2d_homogeneous = K @ camera_pose @ point_3d_homogeneous
        # Convert to non-homogeneous coordinates
        point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]
        # Apply distortion
        if dist_coeffs is not None:
            point_2d = self._apply_distortion(point_2d, dist_coeffs)
        # Return the projected point
        return point_2d
    
    def _project_point_vector_input(self, params, projection_method: ProjectionMethod) -> np.ndarray:
        '''
        Projects a 3D point to 2D.

        Parameters
        ----------
        params : np.ndarray
            Parameters to be passed to _project_3d_point_to_2d (vector).
        projection_method : ProjectionMethod
            Projection method to use.

        Returns
        -------
        point_2d : np.ndarray
            2D point (vector of 2 values).

        Notes:
        ------
        params will have a different format depending on the projection method used.
            for PROJECTION_MATRIX: [x, y, z, p11, p12, p13, p14, p21, p22, p23, p24, p31, p32, p33, p34]
            for PROJECTION_MATRIX_WITH_DISTORTION: [x, y, z, p11, p12, p13, p14, p21, p22, p23, p24, p31, p32, p33, p34, k1, k2, p1, p2, k3]
            for DECOMPOSED_PROJECTION_MATRIX: [x, y, z, fx, fy, cx, cy, r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3]
            for DECOMPOSED_PROJECTION_MATRIX_WITH_DISTORTION: [x, y, z, fx, fy, cx, cy, r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3, k1, k2, p1, p2, k3]
        '''
        # Determine the correct function to use for projection based on the method passed in, and convert the parameters to the correct format
        match projection_method:
            case ProjectionMethod.PROJECTION_MATRIX:
                if len(params) != 15:
                    raise ValueError("Incorrect number of parameters for projection method PROJECTION_MATRIX")
                projected_point_2d = self._project_3d_point_to_2d_via_projection(params[:3], params[3:].reshape((3, 4)))
            case ProjectionMethod.PROJECTION_MATRIX_WITH_DISTORTION:
                if len(params) != 20:
                    raise ValueError("Incorrect number of parameters for projection method PROJECTION_MATRIX_WITH_DISTORTION")
                projected_point_2d = self._project_3d_point_to_2d_via_projection(params[:3], params[3:15].reshape((3, 4)), params[15:])
            case ProjectionMethod.DECOMPOSED_PROJECTION_MATRIX:
                if len(params) != 19:
                    raise ValueError("Incorrect number of parameters for projection method DECOMPOSED_PROJECTION_MATRIX")
                K = np.array([[params[3],         0, params[5]],
                              [        0, params[4], params[6]],
                              [        0,         0,         1]])
                projected_point_2d = self._project_3d_point_to_2d_via_decomposed_projection(params[:3], K, params[7:].reshape((3, 4)))
            case ProjectionMethod.DECOMPOSED_PROJECTION_MATRIX_WITH_DISTORTION:
                if len(params) != 24:
                    raise ValueError("Incorrect number of parameters for projection method DECOMPOSED_PROJECTION_MATRIX_WITH_DISTORTION")
                K = np.array([[params[3],         0, params[5]],
                              [        0, params[4], params[6]],
                              [        0,         0,         1]])
                projected_point_2d = self._project_3d_point_to_2d_via_decomposed_projection(params[:3], K, params[7:19].reshape((3, 4)), params[19:])
            case _:
                raise ValueError("Invalid projection method")
        return projected_point_2d
    
    def _single_point_reprojection_error(self, params, method, point_2d):
        '''
        Calculates the reprojection error for a single point.

        Parameters
        ----------
        params : np.ndarray
            Parameters to be passed to _project_point_vector_input (vector).
        method : ProjectionMethod
            Projection method to use.
        point_2d : np.ndarray
            2D point (vector of 2 values).

        Returns
        -------
        error : float
            Reprojection error.

        Notes:
        ------
        params is a vector of variable length, please see the notes for _project_point_vector_input for the format of params for each projection method.
        point_2d should be of the form: [x, y]
        '''
        # Project point
        projected_point_2d = self._project_point_vector_input(params, method)
        # Calculate error
        error = (point_2d - projected_point_2d)**2
        return error

if __name__ == "__main__":
    intrinsic_matrix = np.array([[925.79882927,   0.,         635.51907178],
                                 [  0.,         923.71342657, 483.87251378],
                                 [  0.,           0.,           1.        ]])
