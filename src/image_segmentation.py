import os
from dataloader import Image
import cv2 as cv
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np

def get_image(image_name):
    # Get Path of images
    path = os.getcwd()
    if path[0] == '/':
        delimiter = '/'
    else:
        delimiter = '\\'
    folders = path.split(delimiter)
    image_folder = folders[0:-1] + ['docs', 'assets', 'test_equations']
    image_folder = delimiter.join(image_folder)
    # Load image
    file_path = image_folder + delimiter + image_name
    img = cv.imread(file_path, cv.IMREAD_COLOR)

    return img

def binarize_img(img):
    # Conver the image to grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Adaptive thresholding to binarize image
    thresh = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 99, 20)

    bw_img = thresh/255
    return bw_img

def convert_2D(bw_img):
    # Convert image to 2D array based on pixel location
    # There is probably a quicker way to do this
    [row, col] = bw_img.shape
    x1 = []
    x2 = []
    for r in range(row):
        for c in range(col):
            if bw_img[r, c] == 0:
                x1.append(c)
                x2.append(-r + row)

    return np.array(x1), np.array(x2)

def convert_to_img(x1, x2, idx):
    # Shift each cluster so they start at 0,0
    x1_offset = x1[idx] - np.min(x1[idx])
    x2_offset = x2[idx] - np.min(x2[idx])

    # Image Edge  Buffer
    buff = 2;

    # Add buffer to the image edges
    Col = np.max(x1_offset) + buff
    Row = np.max(x2_offset) + buff

    if Col > Row:
        img_segment = np.ones([Col, Col])
    else:
        img_segment = np.ones([Row, Row])

    for k in range(len(x1_offset)):
        if Col > Row:
            diff = int((Col-Row)/2)
            img_segment[x2_offset[k]+int(buff/2)+diff, x1_offset[k]+int(buff/2)] = 0
        else:
            diff = int((Row-Col)/2)
            img_segment[x2_offset[k]+int(buff/2), x1_offset[k]+int(buff/2)+diff] = 0

    img_segment = np.flip(img_segment,axis=0)

    img_segment = cv.resize(img_segment, dsize=(45,45))

    return img_segment

def save_image_segment(img_segment, cluster, image_name):
    # Get Path of images
    path = os.getcwd()
    if path[0] == '/':
        delimiter = '/'
    else:
        delimiter = '\\'
    folders = path.split(delimiter)
    image_folder = folders[0:-1] + ['docs', 'assets', 'test_equations']
    image_folder = delimiter.join(image_folder)

    # Load image
    dot = '.'
    file_name = image_name.split(dot)
    # print(file_name[0])
    image_name = dot.join([file_name[0]+str(cluster),file_name[-1]])

    save_name = image_folder + delimiter + image_name
    # print(save_name)
    cv.imwrite(save_name, img_segment*255)


def save_image_coordinates(bl_coord, image_name):
    # Get Path of images
    path = os.getcwd()
    if path[0] == '/':
        delimiter = '/'
    else:
        delimiter = '\\'
    folders = path.split(delimiter)
    image_folder = folders[0:-1] + ['docs', 'assets', 'test_equations']
    image_folder = delimiter.join(image_folder)

    dot = '.'
    file_name = image_name.split(dot)
    # print(file_name[0])
    image_name = dot.join([file_name[0],"csv"])
    save_name = image_folder + delimiter + image_name

    file = open(save_name,'w')
    for coord in bl_coord:
        file.write(str(coord) + "\n")
    file.close()


# matt's additions
def DBSCAN_clustering(img):
    # DBSCAN routine
    bw_img = binarize_img(img)
    x1, x2 = convert_2D(bw_img)
    X = np.stack((x1,x2),axis=1)
    clustering = DBSCAN(eps=10, min_samples=10).fit(X)

    # Arrange clusters from left most to right most
    min_c = []
    for cluster in range(np.max(clustering.labels_)+1):
        idx = np.where(clustering.labels_ == cluster)
        min_c.append(np.min(x1[idx]))

    sorted_segments = sorted(range(len(min_c)), key=lambda k: min_c[k])

    # Convert each cluster back to an image
    sub_images = []
    sub_image_bl_corners = []
    for cluster in range(np.max(clustering.labels_)+1):
        idx = np.where(clustering.labels_ == sorted_segments[cluster])
        image_bl_corner = (int(np.min(x1[idx])), int(img.shape[0] - np.min(x2[idx])))
        img_segment = convert_to_img(x1, x2, idx)

        # save
        sub_images.append(img_segment)
        sub_image_bl_corners.append(image_bl_corner)
    
    return sub_images, sub_image_bl_corners


##############################################################################################

for k in range(5):
    if __name__ == "__main__":
        Eq_name = "Eq4_" + str(k+1)
        image_name = Eq_name + "\\" + Eq_name + ".PNG"
        img = get_image(image_name)
        # Show the original Image
        # cv.imshow("image",img)
        # cv.waitKey()

        # Show the binarized Image
        bw_img = binarize_img(img)
        # cv.imshow("image",bw_img)
        # cv.waitKey()

        # Convert image to two features
        x1, x2 = convert_2D(bw_img)

        # Perform DBSCAN clustering on the two features
        X = np.stack((x1,x2),axis=1)
        clustering = DBSCAN(eps=10, min_samples=10).fit(X)
        # plt.scatter(x1, x2, c=clustering.labels_)
        # plt.axis('equal')
        # plt.show()


        # Arrange clusters from left most to right most
        min_c = []
        for cluster in range(np.max(clustering.labels_)+1):
            idx = np.where(clustering.labels_ == cluster)
            min_c.append(np.min(x1[idx]))

        # Round the vertical pixel locations, we can cluster vertically stacked groups
        min_round = []
        # Vertical proximity
        vp = 25 # pixels
        for c in min_c:
            min_round.append(int(vp * round(float(c)/vp)))
        sorted(min_round)

        # Loop over clusters and pop similar clusters
        unique, ind, count = np.unique(min_round, return_index=True, return_counts=True)
        for k in unique:
            i = np.where(min_round == k)
            if len(i[0]) > 1:
                for c in i[0]:
                    clustering.labels_[clustering.labels_ == c] = i[0][0]

        # Arrange new clusters from left most to right most
        for cluster in range(np.max(clustering.labels_)+1):
            try:
                idx = np.where(clustering.labels_ == cluster)
            except:
                pass


        sorted_segments = sorted(range(len(min_round)), key=lambda k: min_round[k])


        image_bl_corner = []
        # Convert each cluster back to an image
        for cluster in range(len(sorted_segments)):
            try:
                idx = np.where(clustering.labels_ == sorted_segments[cluster])
                image_bl_corner.append((int(np.min(x1[idx])), int(img.shape[0] - np.min(x2[idx]))))

                # plt.scatter(x1[idx], x2[idx])
                # plt.show()

                img_segment = convert_to_img(x1, x2, idx)
                # Save image segment as new image
                save_image_segment(img_segment, cluster, image_name)

            # If image cluster was grouped with another
            except:
                pass

    # Save location of character within image
    save_image_coordinates(image_bl_corner, image_name)
