import cv2
import os
from natsort import natsorted

if __name__ == '__main__':
    # folder_name = "shrunk_cal_images"
    folder_name = "large_cal_images"
    images = natsorted(os.listdir(folder_name)) # read in images
    # cv2.namedWindow('Img',cv2.WINDOW_NORMAL)
    for image_name in images:
        img = cv2.imread(folder_name+"/"+str(image_name))
        print("Image",image_name,"shape is",img.shape)
        #Image should be shape 403(2) x 302(4) for (large)
        if img.shape[0] == 4032 or img.shape[0] == 403:
            flipped_img = cv2.transpose(img)

            cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
            cv2.imshow("Original", img)
            cv2.namedWindow('Flipped', cv2.WINDOW_NORMAL)
            cv2.imshow("Flipped", flipped_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print("New Image shape:",flipped_img.shape)
            cv2.imwrite(folder_name+"/"+str(image_name),flipped_img)