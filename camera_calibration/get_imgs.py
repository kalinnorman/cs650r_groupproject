import cv2

cap = cv2.VideoCapture("calib_vid.mov")

img_cnt = 0

while cap.isOpened():
    success, img = cap.read()
    if success and img_cnt % 20 == 0:
        print("Writing Image",img_cnt)
        cv2.imwrite("calibration_imgs/img"+str(img_cnt)+".png", img)
    elif not success:
        print("Can't receive frame (stream end?). Exiting...")
        break
    img_cnt += 1

cap.release()
cv2.destroyAllWindows()