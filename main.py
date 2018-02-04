import os
import cv2
import funct.label_image.label_image as li
import funct.opt_flow.opt_flow as of
import funct.bg_sub.bg_sub as bgs

import tensorflow as tf
import glob
import image_slicer
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image

files_glob = glob.glob("./in/*.png")


# def pil_grid(images, max_horiz=np.iinfo(int).max):
#     n_images = len(images)
#     n_horiz = min(n_images, max_horiz)
#     h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
#     for i, im in enumerate(images):
#         h, v = i % n_horiz, i // n_horiz
#         h_sizes[h] = max(h_sizes[h], im.size[0])
#         v_sizes[v] = max(v_sizes[v], im.size[1])
#     h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
#     im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
#     for i, im in enumerate(images):
#         im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
#     return im_grid
#
# # Runs Label On Input Image And Generates Confidence Map
# for file in files_glob:
#     tf_image = tf.read_file(file)
#     cv_image = cv2.imread(file)
#     tiles = image_slicer.slice(file, 49, save=False)
#     pil_tiles = []
#     for tile in tiles:
#         numpy_tile = np.array(tile.image)
#         print(li.label_image(numpy_tile.tolist()))
#         if (li.label_image(numpy_tile.tolist()) == "road"):
#             numpy_tile.fill(0)
#         else:
#             numpy_tile.fill(255)
#         pil_tiles.append(Image.fromarray(numpy_tile))
#     combined_file = pil_grid(pil_tiles, 7)
#     combined_file.save('out.png')
#     print(combined_file)
n = 0

# # Runs LK
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.6,
#                        minDistance = 50,
#                        blockSize = 7 )
# lk_params = dict( winSize  = (15,15),
#                   maxLevel = 2,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# color = np.random.randint(0,255,(100,3))
# frame_1 = cv2.imread(files_glob[0])
# old_gray = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
# mask = np.zeros_like(frame_1)
# for file in files_glob:
#     while n < len(files_glob) - 1:
#         img_curr = cv2.imread(files_glob[n+1])
#         img_prev = cv2.imread(files_glob[n])
#         # gftt = cv2.goodFeaturesToTrack(cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY), mask = None, **feature_params)
#         p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
#         result = of.optical_flow(img_curr, img_prev, feature_params, lk_params, frame_1, p0, mask, color)
#         cv2.imwrite('./opt_flow_out/' + str(n)+'.png',result[0])
#         old_gray = result[1]
#         p0 = result[2]
#         print(n)
#         n = n + 1


# Runs BG SUBTRACT
for file in files_glob:
    while n < len(files_glob) - 1:
        img_curr = cv2.imread(files_glob[n])
        result = bgs.bg_subtract(img_curr)
        cv2.imwrite('./opt_flow_out/' + str(n)+'.png',result)
        old_gray = result[1]
        p0 = result[2]
        print(n)
        n = n + 1
