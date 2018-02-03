import cv2
import numpy as np

fgbg = cv2.createBackgroundSubtractorMOG2(100,128,True)
kernel = np.ones((5,5),np.uint8)
# erosion = cv2.erode(img,kernel,iterations = 1)
def bg_subtract(frame):
    fgmask = fgbg.apply(frame)
    opening = cv2.morphologyEx(cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    return opening
