from exploratory_tools import walk_through_images
from dataloader import Image
import visualize as vis

import numpy as np
import cv2 as cv
import seaborn as sns

# # walk through images in a folder
# dir_path = '/nethome/mlamsey3/Documents/Coursework/cs7641-data/archive/batch_1/background_images/'
# walk_through_images(dir_path)

# # file manipulation
# file_path = '/nethome/mlamsey3/Documents/Coursework/cs7641-data/archive/batch_1/background_images/0a9862af-f878-41b9-aef8-7051b737834a.jpg'
# img = Image(file_path)
# print(img.img.shape)
# sub_image_x_y_l_h = [10, 10, 100, 100]
# sub_img = img.get_subimage(sub_image_x_y_l_h)
# print(sub_img.img.shape)

# cv.imshow("image", img.img)
# cv.waitKey()
# cv.imshow("image", sub_img.img)
# cv.waitKey()

img = cv.imread("/Users/mlamsey/Documents/GT/Coursework/cs7641-project/docs/assets/images/Set_2/Equation2.png")
vis.character_clustering(img)
exit()

img = cv.imread("/Users/mlamsey/Documents/GT/Coursework/cs7641-project/docs/assets/images/Set_2/Equation22.png")
import features as u
print(u.intensity(img))
