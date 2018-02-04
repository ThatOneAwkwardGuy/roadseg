import numpy as np
import cv2

def optical_flow(curr_img, prev_img,feature_params,lk_params,first_frame, p0, mask, color):
    while(1):
        frame = curr_img
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        old_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        good_new = p1[st==1]
        good_old = p0[st==1]
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        return [img,old_gray,p0]

        # cv2.imshow('frame',img)
        # k = cv2.waitKey(30) & 0xff
        # if k == 27:
        #     break
    #
    #     old_gray = frame_gray.copy()
    #     p0 = good_new.reshape(-1,1,2)
    # cv2.destroyAllWindows()
    # cap.release()


# def optical_flow(curr_img, prev_img):
#     frame1 = prev_img
#     prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
#     hsv = np.zeros_like(frame1)
#     hsv[...,1] = 255
#     while(1):
#         frame2 = curr_img
#         next = cv2.cvtColor(curr_img,cv2.COLOR_BGR2GRAY)
#         flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 4, 11, 10, 7, 2.4, 0)
#         mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
#         hsv[...,0] = ang*180/np.pi/2
#         hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#         bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
#         return bgr
#         # cv2.imshow('frame2',bgr)
#         # k = cv2.waitKey(30) & 0xff
#         # if k == 27:
#         #     break
#         # elif k == ord('s'):
#         #     cv2.imwrite('opticalfb.png',frame2)
#         #     cv2.imwrite('opticalhsv.png',bgr)
#         # prvs = next
