import cv2
import numpy as np
from tqdm import tqdm

class Disparity():
    def __init__(self, patch_size, d_max):
        self.patch_size = patch_size
        self.d_max = d_max
        return
    
    def compute(self, left_img, right_img, left_img_offset=0):
        '''
        Performs Image Rectification on a left and right image pair. 
        Purpose is to put the image planes parallel to each other with parallel epipolar lines. 
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
        disparity_img = np.zeros((right_img.shape[0],right_img.shape[1]))
        for l_row in tqdm(range(self.patch_size//2 - left_img_offset, h - self.patch_size//2, 1), desc='Disparity Calculations'):
            for l_col in range(self.patch_size//2, w - self.patch_size//2, 1):
                pix_disp = self.compute_patch_match(left_img, right_img, l_row, l_col, left_img_offset)
                disparity_img[l_row,l_col] = pix_disp
        cv2.normalize(disparity_img, disparity_img, 0, 255, cv2.NORM_MINMAX)
        return disparity_img
    
    def compute_patch_match(self, l_img, r_img, row_idx, col_idx, left_img_offset):
        # print(f"l_img.shape: {l_img.shape}, r_img.shape: {r_img.shape}, row_idx: {row_idx}, col_idx: {col_idx}")
        if (self.d_max is None) or (int(col_idx - self.d_max) <= 0):
            start_idx = 0
        else:
            if self.d_max < self.patch_size:
                start_idx = int(col_idx - self.patch_size//2)
            else:
                start_idx = int(col_idx - self.d_max)
        # print(f"patch_sz: {self.patch_size}, start_idx: {start_idx}, d_max: {self.d_max}")
        scan_steps = np.arange(start_idx, col_idx + 1, 1)#patch_sz)
        if len(scan_steps) == 1:
            return 0
        ssd = np.zeros((len(scan_steps),))        
        l_img_patch = l_img[(row_idx+left_img_offset) - self.patch_size//2 : (row_idx+left_img_offset) + np.ceil(self.patch_size/2).astype(int), 
                            col_idx - self.patch_size//2 : col_idx + np.ceil(self.patch_size/2).astype(int)]
        wL = l_img_patch.reshape((self.patch_size*self.patch_size,))
        cnt = 0
        for col_scanline in range(start_idx, col_idx + 1, 1):#patch_sz):
            d = col_idx - col_scanline
            if (col_scanline - self.patch_size//2) < 0:
                sub_col_section = col_scanline + np.ceil(self.patch_size/2).astype(int)
                r_img_patch = r_img[row_idx - self.patch_size//2 : row_idx + np.ceil(self.patch_size/2).astype(int), 
                                0 : sub_col_section]
                wR = r_img_patch.reshape((self.patch_size*sub_col_section,))
                
                wL_cropped = l_img[row_idx - self.patch_size//2 : row_idx + np.ceil(self.patch_size/2).astype(int), 
                            col_idx - (sub_col_section//2) : col_idx + np.ceil(sub_col_section/2).astype(int)].reshape((self.patch_size*sub_col_section,))
                ssd[cnt] += np.linalg.norm(wL_cropped - wR,2)**2
            else:
                r_img_patch = r_img[row_idx - self.patch_size//2 : row_idx + np.ceil(self.patch_size/2).astype(int), 
                                col_scanline - self.patch_size//2 : col_scanline + np.ceil(self.patch_size/2).astype(int)]
                wR = r_img_patch.reshape((self.patch_size*self.patch_size,))
                ssd[cnt] += np.linalg.norm(wL - wR,2)**2
            cnt += 1
        wta_idx = np.argmax(ssd) # winner-takes-all
        disp = col_idx - scan_steps[wta_idx]
        assert(disp >= 0) # Must be non-negative
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