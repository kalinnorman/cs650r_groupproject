import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Disparity():
    def __init__(self, patch_size, d_max):
        self.patch_size = patch_size
        self.d_max = d_max
        return
    
    def compute(self, left_img, right_img, left_img_offset=0):
        '''
        Performs Patch Matching algorithm to build a disparity map of the image. 
        Input:
            left_img: left image of object
            right_img: image where object appears to have shifted left
        Output:
            disparity_img: disparity image normalized between (0,255)
        '''
        # Asserts same size images
        assert(left_img.shape == right_img.shape)
        assert(len(left_img.shape) == 2) # 2D gray image
        w, h = left_img.shape[1], left_img.shape[0]
        assert(w > (self.d_max + self.patch_size))
        assert(h > self.patch_size)
        disparity_img = np.zeros((right_img.shape[0],right_img.shape[1]))
        for l_row in tqdm(range(self.patch_size//2, h - self.patch_size//2, 1), desc='Disparity Calculations'):
            for l_col in range(self.d_max + self.patch_size//2, w - self.patch_size//2, 1):
            # for l_col in range(self.patch_size//2, w - self.patch_size//2, 1):
                pix_disp = self.compute_patch_match(left_img, right_img, l_row, l_col, left_img_offset)
                disparity_img[l_row,l_col] = pix_disp
        # cv2.normalize(disparity_img, disparity_img, 0, 255, cv2.NORM_MINMAX)
        return disparity_img
    
    def compute_patch_match(self, l_img, r_img, row_idx, col_idx, left_img_offset):
        # Search range in the right image
        start_idx = int(col_idx - self.d_max)
        scan_steps = np.arange(start_idx, col_idx + 1, 1)
        # x_min = max(self.patch_size//2, col_idx - self.d_max)
        # x_max = min(l_img.shape[1] - self.patch_size//2, col_idx + self.d_max)
        # scan_steps = np.arange(x_min,x_max)
        
        l_img_patch = l_img[row_idx - self.patch_size//2 : row_idx + np.ceil(self.patch_size/2).astype(int), 
                            col_idx - self.patch_size//2 : col_idx + np.ceil(self.patch_size/2).astype(int)]
        wL = l_img_patch.reshape((self.patch_size*self.patch_size,))
        cnt = 0
        ssd = np.zeros((len(scan_steps),))
        for col_scanline in scan_steps:
            # d = col_idx - col_scanline
            r_img_patch = r_img[row_idx - self.patch_size//2 : row_idx + np.ceil(self.patch_size/2).astype(int), 
                            col_scanline - self.patch_size//2 : col_scanline + np.ceil(self.patch_size/2).astype(int)]
            wR = r_img_patch.reshape((self.patch_size*self.patch_size,))
            ssd[cnt] = np.linalg.norm((wL - wR)**2,2)
            cnt += 1
        wta_idx = np.argmin(ssd) # winner-takes-all
        # plt.figure()
        # plt.plot(scan_steps,ssd)
        # print(scan_steps[wta_idx])
        # plt.show()
        disp = abs(col_idx - scan_steps[wta_idx])
        # assert(disp >= 0) # Must be non-negative
        return disp
    
    def compute_disparity_cgpt(self, imgL, imgR):#, block_size=5, disparity_range=16):
        """
        Compute disparity map using basic block matching algorithm.

        :param imgL: Left image (numpy array) grayscale.
        :param imgR: Right image (numpy array) grayscale.
        :return: Disparity map.
        """

        # Image dimensions
        h, w = imgL.shape

        # Padding
        pad = self.patch_size // 2

        # Initialize disparity map
        disparity_map = np.zeros_like(imgL)

        # Iterate over each pixel in the left image
        for y in range(pad, h - pad):
            for x in range(pad, w - pad):
                # Define the patch in the left image
                block_left = imgL[y - pad:y + pad + 1, x - pad:x + pad + 1]

                # Search range in the right image
                x_min = max(pad, x - self.d_max)
                x_max = min(w - pad, x + self.d_max)

                # Initialize the best match score and position
                best_score = float('inf')
                best_x = 0

                # Iterate over each position in the search range
                for xR in range(x_min, x_max):
                    # Define the patch in the right image
                    block_right = imgR[y - pad:y + pad + 1, xR - pad:xR + pad + 1]

                    # Compute the sum of squared differences (SSD)
                    ssd = np.sum((block_left - block_right) ** 2)

                    # Update best match if a better score is found
                    if ssd < best_score:
                        best_score = ssd
                        best_x = xR

                # Compute disparity
                disparity_map[y, x] = abs(x - best_x)

        return disparity_map