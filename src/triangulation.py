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
        
def triangulate_lm(K, R1, t, R2, t, pts1, pts2, dist_params=None):
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
        
        '''
        