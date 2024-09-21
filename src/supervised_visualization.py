import numpy as np
import os
import PIL
import pandas as pd
import glob
import pickle
from tqdm import tqdm
from PIL import Image
from os.path import exists
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import cv2 as cv
import image_identification as iden



est = iden.image_id(load_dataframe = True, load_models=True ,predict_only=False)
path = os.getcwd()
delimiter = '\\'
folders = path.split(delimiter)
directory_images = folders[0:-1] + ['cs7641-project','docs', 'assets', 'sample_images']
directory_images = delimiter.join(directory_images)

model_directory = folders[0:-1]+['cs7641-project','docs','assets','models']
model_directory = delimiter.join(model_directory)

saved_images_on_pc = ['C:','data','extracted_images']
saved_images_on_pc = delimiter.join(saved_images_on_pc)

data_name = 'full_dataset'
dataset_name = data_name+'.sav'
models = [data_name+'GNB_classifier.sav',data_name+'dt_classifier.sav',data_name+'sgd_classifier.sav']

pr = pickle.load(open(model_directory+'\\'+dataset_name,'rb'))
X_train, X_test, y_train, y_test = train_test_split(pr.iloc[:,:-1], pr.iloc[:,-1], test_size=0.3, shuffle=False)

classifiers = []
for i in range(len(models)):
    print(i)
    filename = os.path.join(model_directory+'\\'+models[i])
    classifier = pickle.load(open(filename,'rb'))
    classifiers.append(classifier)
    print(classifier)
    #display = metrics.roc_auc_score(y_test,classifiers[i].predict_proba(X_test), multi_class='ovr')

    y_pred = classifier.predict(X_test)
    y_test_0 = y_test.values
    for i in range(len(y_pred)):
        if y_pred[i]>0:
            y_pred[i] = 1
        if y_test_0[i] >0:
            y_test_0[i] = 1


    y_pred = 1-y_pred
    y_test_0 = 1-y_test_0

    print(y_pred)
    print(y_test_0)
    svc_disp = metrics.RocCurveDisplay.from_predictions(y_test_0,y_pred)
    plt.show()
    #display = metrics.roc_auc_score(y_test,classifier.predict_proba(X_test), multi_class='ovr')




