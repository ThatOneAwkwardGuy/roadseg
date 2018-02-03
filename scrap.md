import os
import cv2
import funct.label_image.label_image as li
import funct.opt_flow.opt_flow as of
import tensorflow as tf
import glob
import image_slicer
import numpy as np

# print(glob.glob("./in/*.png"))

files_glob = glob.glob("./in/*.png")

# n = 0
# while n < len(files_glob) - 1:
#     image = cv2.imread(files_glob[n+1])
#     img_prev = cv2.imread(files_glob[n])
#     result = of.optical_flow(image, img_prev)
#     cv2.imwrite('./opt_flow_out/' + str(n)+'.png',result)
#     n = n + 1

for file in files_glob:
    image_cv2 = cv2.imread(file)
    # image_slicer.slice(file, 28)
    image = tf.read_file(file)
    label = li.label_image(image)
    # 
    if(label == "road"):
        # print("hello")
        img = np.zeros(image.shape,dtype=np.uint8)
        img.fill(255) # or img[:] = 255
        cv2.imwrite(file,img)
    else:
        cv2.imwrite(file,np.ones(image.shape,dtype=np.uint8))
    print(file)
    
    
# for file in files_glob:
#     image = cv2.imread("in/B.png")
#     img_prev = cv2.imread("in/A.png")

# directory = os.fsencode("in")
# for file in os.listdir(directory):
#     filename = os.fsdecode(file)
#     if filename.endswith(".png"):
#         path = "in/" + filename
#         # print(path)
#         # image = tf.read_file(path, "frame")
#         image = cv2.imread("in/B.png")
#         img_prev = cv2.imread("in/A.png")
#         # print(image)
#         # li.label_image(image)
#         result = of.optical_flow(image, img_prev)
#         cv2.imwrite('messigray.png',result)
#         # print(result)
#         # cv2.imshow('frame',image)
