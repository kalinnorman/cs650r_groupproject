import cv2 as cv
import numpy as np

class FeatureHelper():
    def __init__(self):
        self.sift = cv.SIFT_create()
        self.bfm = cv.BFMatcher()
        self.matches = dict()
        return
    
    def compute_matches(self, imgs):
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
                    if key in self.matches:
                        self.matches[(prev_img_name, curr_img_name)].append(
                        (kp_p[m.queryIdx].pt, kp_c[m.trainIdx].pt))
                    else:
                        self.matches[(prev_img_name, curr_img_name)] = [
                            (kp_p[m.queryIdx].pt, kp_c[m.trainIdx].pt)
                        ]

            # Update prev variables
            prev_img_name = curr_img_name
            prev_img = curr_img    
        return self.matches
