import cv2
import numpy as np

def color_to_hsv(img):
    hsvimage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsvimage

def color_to_greyscale(img):
    greyimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return greyimage

def apply_white_mask(img):
    mask_white = cv2.inRange(img, 200, 255)
    return mask_white

def apply_yellow_mask(img):
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(img, lower_yellow, upper_yellow)
    return mask_yellow

def lane_marking_filter(white_mask, yellow_mask,grey_image):
    yellow_or_white_mask = cv2.bitwise_or(white_mask, yellow_mask)
    yellow_or_white_and_grey_mask = cv2.bitwise_and(grey_image, yellow_or_white_mask)
    yellow_or_white_and_grey_mask_blur = cv2.GaussianBlur(yellow_or_white_and_grey_mask,(5,5),0)
    return yellow_or_white_and_grey_mask

def canny_edges(img):
    low_threshold = 50
    high_threshold = 150
    canny_edges = cv2.Canny(img,low_threshold,high_threshold)
    black_image = np.zeros(img.shape).astype(img.dtype)
    cv2.fillPoly(black_image, [np.array([[600, 180], [300, 375], [800, 375], [600, 180]])], [255,255,255])
    result = cv2.bitwise_and(canny_edges, black_image)
    return canny_edges

def hough_lines(img):
    canny_img = img
    black_image = np.zeros(img.shape).astype(img.dtype)
    hough_lines = cv2.HoughLinesP(canny_img,1, np.pi / 180, 0, None, 0, 0)
    for line in hough_lines:
        for x1,y1,x2,y2 in line:
            cv2.line(black_image,(x1,y1),(x2,y2),(255,0,0),2,15)
    return black_image

def process_one_frame(image):
    hsv_image = color_to_hsv(image)
    greyscale_image = color_to_greyscale(image)
    white_mask = apply_white_mask(greyscale_image)
    yellow_mask = apply_yellow_mask(hsv_image)
    lane_markings = lane_marking_filter(white_mask,yellow_mask,greyscale_image)
    canny_edges_img = canny_edges(greyscale_image)
    hough_lines_img = hough_lines(canny_edges_img).astype(image.dtype)
    final_image = cv2.bitwise_or(greyscale_image,hough_lines_img)
    return canny_edges_img
