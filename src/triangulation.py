import numpy as np

def triangulate_lindstrom(K, E, R, img_pts1, img_pts2):
    """
    Triangulate points in two images using the camera matrices and the
    corresponding image points. The method herein is based on the paper:
    "Triangulation Made Easy" by Peter Lindstrom in 2010.

    Parameters
    ----------
    K : numpy.ndarray (size (3,3))
        Camera matrix.
    E : numpy.ndarray (size (3,3))
        Essential matrix, which includes the rotation and translation between
        the two cameras.
    R : numpy.ndarray (size (3,3))
        Rotation matrix between the two cameras.
    img_pts1 : numpy.ndarray (size (N,2))
        Image points in the first image.
    img_pts2 : numpy.ndarray (size (N,2))
        Image points in the second image.
        
    Returns
    -------
    pts_3d : numpy.ndarray (size (N,3))
        Triangulated 3D points.
    """
    # Identify helpful values
    num_pts = img_pts1.shape[0]
    if num_pts != img_pts2.shape[0]:
        raise ValueError('The number of image points in each image must be the same.')

    # Convert image points to homogeneous coordinates
    img_pts1_homogenous = np.hstack((img_pts1, np.ones((img_pts1.shape[0], 1))))
    img_pts2_homogenous = np.hstack((img_pts2, np.ones((img_pts2.shape[0], 1))))

    # Normalize image points
    K_inv = np.linalg.inv(K)
    img_pts1_norm = K_inv @ img_pts1_homogenous.T
    img_pts2_norm = K_inv @ img_pts2_homogenous.T

    # S matrix defined by equation (4) in the paper
    S = np.array([[1, 0, 0], [0, 1, 0]])

    # Define the niter1 method outlined in listing 3 of the paper
    def niter1(x, x_prime, E):
        '''
        Parameters
        ----------
        x : numpy.ndarray (size (3,1))
            Normalized homogeneous coordinates of a point in the first image.
        x_prime : numpy.ndarray (size (3,1))
            Normalized homogeneous coordinates of the corresponding point in the second image.
        E : numpy.ndarray (size (3,3))
            Essential matrix.
        '''
        # Verify inputs
        if x.shape != (3,1):
            raise ValueError('x must be a column vector of size (3,1).')
        if x_prime.shape != (3,1):
            raise ValueError('x_prime must be a column vector of size (3,1).')
        if E.shape != (3,3):
            raise ValueError('E must be a matrix of size (3,3).')
        # Compute E_tilda for convenience (equation (5) in the paper)
        E_tilda = S @ E @ S.T
        # Calculations as defined in listing 3 of the paper
        n = S @ E @ x_prime
        n_prime = S @ E.T @ x
        a = n.T @ E_tilda @ n_prime
        b = 0.5 * (n.T @ n + n_prime.T @ n_prime)
        c = x.T @ E @ x_prime
        d = np.sqrt(b**2 - a*c)
        lambd = c / (b + d)
        delta_x = lambd * n
        delta_x_prime = lambd * n_prime
        n = n - E_tilda @ delta_x_prime
        n_prime = n_prime - E_tilda.T @ delta_x
        delta_x = ((delta_x.T @ n) / (n.T @ n)) * n
        delta_x_prime = ((delta_x_prime.T @ n_prime) / (n_prime.T @ n_prime)) * n_prime
        x = x - S.T @ delta_x
        x_prime = x_prime - S.T @ delta_x_prime
        return x, x_prime

    # Estimate the 3D points
    pts_3d = np.zeros((num_pts, 3))
    for i in range(num_pts):
        # Correct the image points via method in the paper
        x = img_pts1_norm[:,i].reshape(3,1)
        x_prime = img_pts2_norm[:,i].reshape(3,1)
        x, x_prime = niter1(x, x_prime, E)
        # Solve for the 3D point estimate using equation (13) in the paper
        z = np.array([[0, -x[2,0], x[1,0]], [x[2,0], 0, -x[0, 0]], [-x[1,0], x[0,0], 0]]) @ R @ x_prime
        X = ((z.T @ E @ x_prime) / (z.T @ z)) * x
        pts_3d[i,:] = X.T
    
    # return the 3D points
    return pts_3d
        
def triangulate_lm(K, R1, t1, R2, t2, pts1, pts2, dist_params=None):
    '''
    Triangulate points in the two images using the projection matrices and the
    image coordinates of the matching points in the two images. 
    Triangulation is performed using the Levenberg-Marquardt algorithm.
    Distortion paramters can optionally be included if the images were
    not previously undistorted.
    
    Parameters
    ----------
    K : numpy.ndarray (size (3,3))
        Camera intrinsics matrix.
    R1 : numpy.ndarray (size (3,3))
        Rotation matrix for the first camera.
    t1 : numpy.ndarray (size (3,1))
        Translation vector for the first camera.
    R2 : numpy.ndarray (size (3,3))
        Rotation matrix for the second camera.
    t2 : numpy.ndarray (size (3,1))
        Translation vector for the second camera.
    pts1 : numpy.ndarray (size (N,2))
        Image points in the first image.
    pts2 : numpy.ndarray (size (N,2))
        Image points in the second image.
    dist_params : numpy.ndarray (size (5,) or 'None')
        Distortion parameters for the camera.
        If 'None', then undistorted image points are assumed.
    '''
    def obj_func(optimization_params, other_variables, expected_vals):
        '''
        This is a residual function.
        Calculates the difference between the measured image points and the
        projected image points.
        
        Parameters
        ----------
        optimization_params : numpy.ndarray (size (M,))
            Vector of optimization parameters, which are the 3D points
        other_variables : list 
            :ist of other variables needed to calculate the residual.
            This includes K, R1, t1, R2, t2 and optionally dist_params.
        expected_vals : numpy.ndarray (size (2*N,))
            Vector of expected values (i.e. the measured image points).
        '''
        def apply_distortion(point_2d, dist_coeffs, K) -> np.ndarray:
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

        # Unpack the optimization parameters
        pts_3d = optimization_params.reshape(-1,3)
        # Unpack the other variables
        K = other_variables[:9].reshape(3,3)
        R1 = other_variables[9:18].reshape(3,3)
        t1 = other_variables[18:21].reshape(3,1)
        R2 = other_variables[21:30].reshape(3,3)
        t2 = other_variables[30:33].reshape(3,1)
        if len(other_variables) > 33:
            dist_params = other_variables[33:]
            if len(dist_params) != 5:
                raise ValueError(f'dist_params had length {len(dist_params)}, when it should have been 5.')
        else:
            dist_params = None
        # Unpack the expected values
        pts1 = expected_vals[:len(expected_vals)//2].reshape(-1,2)
        pts2 = expected_vals[len(expected_vals)//2:].reshape(-1,2)

        if len(pts1) != len(pts2):
            raise ValueError('pts1 and pts2 must have the same length.')
        if len(pts1) != len(pts_3d):
            raise ValueError('pts1 and pts_3d must have the same length.')
        
        # Form the projection matrices
        P1 = K @ np.hstack((R1, t1))
        P2 = K @ np.hstack((R2, t2.reshape(3,1)))

        # Calculate residual vector
        resids = np.array([])
        for i in range(len(pts1)):
            # Project the 3D point into the first image
            pt1 = P1 @ np.hstack((pts_3d[i,:], 1))
            pt1 = pt1[:2] / pt1[2]
            # Apply distortion if necessary
            if dist_params is not None:
                pt1 = apply_distortion(pt1, dist_params, K)
            # Calculate the squared residual and append to the array
            resids = np.append(resids, (pts1[i,:] - pt1)**2)
        for i in range(len(pts2)):
            # Project the 3D point into the second image
            pt2 = P2 @ np.hstack((pts_3d[i,:], 1))
            pt2 = pt2[:2] / pt2[2]
            # Apply distortion if necessary
            if dist_params is not None:
                pt2 = apply_distortion(pt2, dist_params, K)
            # Calculate the squared residual and append to the array
            resids = np.append(resids, (pts2[i,:] - pt2)**2)
        # Return the result
        return resids
    
    # Form the initial guess for the 3D points
    pts_3d = np.hstack((np.random.choice([1, 2, -1, -2], (len(pts1), 1)), np.random.choice([1, 2, -1, -2], (len(pts1), 1)), 100*np.ones((len(pts1), 1))))
    optimization_params = pts_3d.ravel()
    # Form the other variables
    if dist_params is not None:
        other_variables = np.hstack((K.ravel(), R1.ravel(), t1.ravel(), R2.ravel(), t2.ravel(), dist_params.ravel()))
    else:
        other_variables = np.hstack((K.ravel(), R1.ravel(), t1.ravel(), R2.ravel(), t2.ravel()))
    # Form the expected values
    expected_vals = np.hstack((pts1.ravel(), pts2.ravel()))
    
    '''
    Find the solution via Levenberg-Marquardt
    '''
    
    def get_jacobian(optimization_params, other_variables, expected_vals):
        '''
        Calculate the jacobian matrix via central differences.
        '''
        # Calculate the jacobian via central differences
        J = np.zeros((len(expected_vals), len(optimization_params)))
        delta = 1e-6
        for i in range(len(optimization_params)):
            opt_temp = optimization_params.copy()
            # Calculate the residual with the positive delta
            opt_temp[i] = optimization_params[i] + delta
            resids_pos = obj_func(opt_temp, other_variables, expected_vals)
            # Calculate the residual with the negative delta
            opt_temp[i] = optimization_params[i] - delta
            resids_neg = obj_func(opt_temp, other_variables, expected_vals)
            # Calculate the jacobian
            J[:,i] = (resids_pos - resids_neg) / (2*delta)
        # Return the jacobian
        return J

    # Perform optimization
    keep_going = True
    the_lambda = 10000
    change_in_resids = []
    going_uphill = 0
    the_I = np.eye(len(optimization_params))
    while keep_going:
        print('in loop')
        # Calculate the residual vector
        resids = obj_func(optimization_params, other_variables, expected_vals)
        print('got resids')
        # Calculate the jacobian matrix
        J = get_jacobian(optimization_params, other_variables, expected_vals)
        print('got jacobian')
        if len(change_in_resids) == 0:
            optimization_params = np.linalg.inv(J.T @ J) @ J.T @ resids
            change_in_resids.append(np.linalg.norm(obj_func(optimization_params, other_variables, expected_vals)) - np.linalg.norm(resids))
            continue
        # Calculate the update vector
        part_a = J.T @ J
        part_c = J.T @ resids
        print(f'resid norm to beat: {np.linalg.norm(resids)}')
        best_resid = []
        lambda_vals = []
        for i in range(100):
            delta = np.linalg.inv(part_a + the_lambda * the_I) @ part_c
            # Update the optimization parameters
            test_optimization_params = optimization_params - delta
            # Check if the update results in an uphill step
            resids_new = obj_func(test_optimization_params, other_variables, expected_vals)
            best_resid.append(np.linalg.norm(resids_new))
            lambda_vals.append(the_lambda)
            print(f'resid norm at step {i}: {best_resid[-1]}')
            if best_resid[-1] > np.linalg.norm(resids):
                # If so, then increase the lambda
                if the_lambda < 1e12:
                    the_lambda = the_lambda * 3
                else:
                    # If the lambda is already maxed out, then pick the best one we had til now and move to the next step
                    the_lambda = lambda_vals[np.argmin(best_resid)]
                    delta = np.linalg.inv(part_a + the_lambda * the_I) @ part_c
                    resids_new = obj_func(optimization_params - delta, other_variables, expected_vals)
                    i = None
                    break
            else:
                # Otherwise, decrease the lambda and break
                if the_lambda > 1e-12:
                    the_lambda = the_lambda / 3
                i = None
                break
        if i is None:
            optimization_params = optimization_params - delta
            # Check if the residual is small enough
            if np.linalg.norm(resids_new) < 1e-6:
                keep_going = False
            # Check if the residual is changing over time enough
            change_in_resids.append(np.linalg.norm(resids_new) - np.linalg.norm(resids))
            if len(change_in_resids) > 10:
                change_in_resids.pop(0)
                if np.all(change_in_resids) < 1e-12:
                    keep_going = False
        else:
            # Break because we only ever increased the residual
            raise ValueError('Failed to find optimization step that decreased the residual.')
        print(f'NEW STEP with dist {np.linalg.norm(resids_new)}')
    # Return the optimized 3D points
    return optimization_params.reshape(-1,3)



        