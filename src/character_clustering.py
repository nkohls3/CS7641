import os
import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
import skimage.measure
import matplotlib.pyplot as plt

import features
import image_segmentation

# Get Path of images
path = os.getcwd()
if path[0] == '/':
    delimiter = '/'
else:
    delimiter = '\\'
folders = path.split(delimiter)
image_folder = folders[0:-1] + ['docs', 'assets', 'images']
image_folder = delimiter.join(image_folder)

# Get all the filenames from a selected folder
filepath = image_folder + delimiter + "N_o" + delimiter
for (root,dirs,files) in os.walk(filepath):
    pass

images = []
# for image_name in image_names:
for file in files:
    img = cv.imread(filepath + file, 0)
    # cv.imshow("image", img)
    # cv.waitKey()
    # Add border to image to help erosion in next step (use same kernel size)
    kernel_size = 9
    top, bottom, left, right = [kernel_size] * 4
    img_border = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=255)
    # Reduce the image size by dilating the black text and resizing using nearest values
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    img = cv.erode(img_border, kernel)
    # cv.imshow("image", img)
    # cv.waitKey()
    # img = cv.resize(img, dsize=(10,10), interpolation=cv.INTER_NEAREST_EXACT)
    # Remove the added border
    # img = img[1:-1,1:-1]
    # cv.imshow("image", img)
    # cv.waitKey()
    images.append(cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1])

# Compute features for each image.
X = np.empty((0, 5))
for image in images:
    x1 = features.intensity(image)
    x2 = features.horizontal_symmetry(image)
    x3 = features.vertical_symmetry(image)
    x4 = features.area_perimeter_ratio(image)
    x5 = features.height_width_ratio(image)
    X = np.vstack((X, [x1, x2, x3, x4, x5]))

# Cluster characters using K-means
# Cluster "N" and "n"
kmeans = KMeans(n_clusters=2, random_state=42)
# Test and plot selected features
feature_labels = ['Intensity', 'Horizontal Symmetry', 'Vertical Symmetry',
                  'Area to Perimeter Ratio', 'Height to Width Ratio']
f1 = 0
f2 = 2
X_sampled = X[:,[f1,f2]]

kmeans = kmeans.fit(X_sampled)
print(kmeans.labels_)

actual_labels = np.array((0,0,1,0,0,0,
                         1,1,0,0,1,0,
                         1,0,0,0,0,0,
                         1,1,1,1,0,1,
                         0,1,1,0,0,0,
                         1,1,0,0,1,1,
                         0,0,0,0,0,1,
                         0,0,0,0,0,0,
                         0,0,0,0,0,0,
                         0,0,0,0,1,0,
                         1,0,1,1,1,0,
                         1,1,0,1,1,1,
                         0,0,0,1,0,0,
                         0,0,0,0,0,1,
                         1,1,1,1,1,1,
                         1,1,1,1,1,1,
                         1,1,1,1))


idx1 = np.where(actual_labels == kmeans.labels_)
idx2 = np.where(-1*actual_labels+1 == kmeans.labels_)

if len(idx1[0]) < len(idx2[0]):
    print(len(idx2[0]),'Correct out of 100')
    actual_labels = -1*actual_labels+1
else:
    print(len(idx1[0]),'Correct out of 100')

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(X_sampled[:,0], X_sampled[:,1], c = kmeans.labels_, cmap = 'bwr')
plt.axis('equal')
plt.xlabel(feature_labels[f1])
plt.ylabel(feature_labels[f2])
plt.title('Predicted Clustering')

plt.subplot(1,2,2)
plt.scatter(X_sampled[:,0], X_sampled[:,1], c = actual_labels, cmap = 'bwr')
plt.axis('equal')
plt.xlabel(feature_labels[f1])
plt.ylabel(feature_labels[f2])
plt.title('Actual Clustering')

plt.show()

#  Clustering on all features
kmeans_all = KMeans(n_clusters=2, random_state=42)
kmeans_all = kmeans_all.fit(X)
print(kmeans_all.labels_)
idx = np.where(actual_labels == kmeans_all.labels_)
print(len(idx[0]),'Correct out of 100')