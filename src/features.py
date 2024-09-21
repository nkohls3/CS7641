import numpy as np
import cv2 as cv

# Image processing functions
def img_scharr_x(img):
    return cv.Scharr(img, ddepth=cv.CV_8U, dx=1, dy=0)

def img_scharr_y(img):
    return cv.Scharr(img, ddepth=cv.CV_8U, dx=0, dy=1)

def img_sobel(img, dx=1, dy=1, ksize=3):
    return cv.Sobel(img, ddepth=cv.CV_8U, dx=dx, dy=dy, ksize=ksize)

# Master feature vectors
def all_features(img):
    features = []
    labels = []

    # bulk features
    (_scharr_features, _scharr_labels) = scharr_features(img)
    features += _scharr_features
    labels += _scharr_labels
    (_sobel_features, _sobel_labels) = sobel_features(img)
    features += _sobel_features
    labels += _sobel_labels

    # rest of the features
    features.append(intensity(img))
    features.append(horizontal_symmetry(img))
    features.append(vertical_symmetry(img))
    # features.append(area_perimeter_ratio(img))
    features.append(height_width_ratio(img))


    labels += [
        "intensity",
        "horizontal_symmetry",
        "vertical_symmetry",
        "height_width_ratio"
    ]

    return np.array(features), labels

def scharr_features(img):
    features = []
    features += scharr_intensity(img)
    features += scharr_vertical_symmetry(img)
    features += scharr_horizontal_symmetry(img)

    labels = [
        "scharr_intensity_x",
        "scharr_intensity_y",
        "scharr_vertical_symmetry_x",
        "scharr_vertical_symmetry_y",
        "scharr_horizontal_symmetry_x",
        "scharr_horizontal_symmetry_y"
    ]

    return features, labels

def sobel_features(img):
    features = []
    features.append(sobel_intensity(img))
    features.append(sobel_vertical_symmetry(img))
    features.append(sobel_horizontal_symmetry(img))

    labels = [
        "sobel_intensity",
        "sobel_vertical_symmetry",
        "sobel_horizontal_symmetry"
    ]

    return features, labels

# Features
def intensity(img):
    # imgray = cv.cvtColor(~img, cv.COLOR_BGR2GRAY)
    return np.sum(img == 0) / np.size(img)

def horizontal_symmetry(img):
    # imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgray_flipped = np.fliplr(img)
    return np.sum(img == imgray_flipped)/np.size(img)

def vertical_symmetry(img):
    # imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgray_flipped = np.flipud(img)
    return np.sum(img == imgray_flipped) / np.size(img)

def area_perimeter_ratio(img):
    # imgray = cv.cvtColor(~img, cv.COLOR_BGR2GRAY)
    contour, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    area = cv.contourArea(contour[0])
    perimeter = cv.arcLength(contour[0], True)
    return (4 * np.pi * area) / (perimeter ** 2)

def height_width_ratio(img):
    # imgray = cv.cvtColor(~img, cv.COLOR_BGR2GRAY)
    blob_mask = np.argwhere(img == 0)
    height = max(blob_mask[:, 0]) - min(blob_mask[:, 0]) + 1
    width = max(blob_mask[:, 1]) - min(blob_mask[:, 1]) + 1
    return height/width

def scharr_intensity(img):
    return (intensity(img_scharr_x(img)), intensity(img_scharr_y(img)))

def scharr_vertical_symmetry(img):
    return (vertical_symmetry(img_scharr_x(img)), vertical_symmetry(img_scharr_y(img)))

def scharr_horizontal_symmetry(img):
    return (horizontal_symmetry(img_scharr_x(img)), horizontal_symmetry(img_scharr_y(img)))

def sobel_intensity(img, dx=1, dy=1, ksize=3):
    return intensity(img_sobel(img, dx, dy, ksize))

def sobel_vertical_symmetry(img, dx=1, dy=1, ksize=3):
    return vertical_symmetry(img_sobel(img, dx, dy, ksize))

def sobel_horizontal_symmetry(img, dx=1, dy=1, ksize=3):
    return horizontal_symmetry(img_sobel(img, dx, dy, ksize))
