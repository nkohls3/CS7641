from copy import copy
import cv2 as cv

class Image:
    def __init__(self, file_path):
        self.file_path = file_path
        self.img = cv.imread(file_path, cv.IMREAD_COLOR)

    def show_waitkey(self):
        cv.imshow("image", self.img)
        return cv.waitKey()

    def get_subimage(self, rectangle_shape):
        if len(rectangle_shape) != 4:
            print("Image::get_subimage: rectangle_shape not length 4.")
            return None

        # get rectangle
        x, y, dx, dy = rectangle_shape
        dx += x
        dy += y

        # build sub image
        sub_img = copy(self)
        sub_img.img = self.img[x:dx, y:dy]
        return sub_img
