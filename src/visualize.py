import numpy as np
import cv2 as cv
import seaborn as sns

from image_segmentation import DBSCAN_clustering, convert_2D

def character_clustering(img):
    sub_images, sub_image_bl_corners = DBSCAN_clustering(img)

    colors = sns.color_palette("bright", len(sub_images))

    for i in range(len(sub_images)):
        sub_image = sub_images[i]
        (x, y) = sub_image_bl_corners[i]
        color = np.array(colors[i]) * 255

        # color image
        # cv.drawMarker(img, (x, y), color=(0,0,255), markerType=cv.MARKER_CROSS, thickness=2)
        img_buffer = 5
        (x1, x2) = convert_2D(sub_image)
        sub_img_x = y - x2 + img_buffer
        sub_img_y = x1 + x - img_buffer
        img[sub_img_x, sub_img_y] = color

    cv.imshow("image", img)
    cv.waitKey()