import cv2
import numpy as np

fgbg = cv2.createBackgroundSubtractorMOG2(300,32,False)
kernel = np.ones((5,5),np.uint8)
# erosion = cv2.erode(img,kernel,iterations = 1)
def bg_subtract(frame):
    resized_image = cv2.resize(frame, (696,256))
    resized_and_blured = cv2.blur(resized_image,(4,4))
    fgmask = fgbg.apply(resized_image)
    bg_subtracted = cv2.morphologyEx(cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    (_, cnts, _) = cv2.findContours(bg_subtracted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_image = resized_image.copy()
    for c in cnts:
        rect = cv2.boundingRect(c)
        if rect[2] < 30 or rect[3] < 30: continue
        # print cv2.contourArea(c)
        x,y,w,h = rect
        cv2.rectangle(final_image,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.putText(final_image,'',(x+w+10,y+h),0,0.3,(0,255,0))
    # cv2.drawContours(final_image, cnts, -1, (0, 255, 0), 3)
    # print(bg_subtracted_contours)
    return final_image
