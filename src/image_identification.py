#!/usr/bin/env python
# coding: utf-8
# Ask Sebastian any questions
# In[75]:

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

class image_id:
    def __init__(self,load_dataframe = True,load_models = False,predict_only = False):
        self.load_models = load_models
        self.load_dataset = load_dataframe
        self.predict_only = predict_only
    def create_dataframe(self,image_directory,save_directory,filename):
        data = []
        label = []
        label_dict = {}
        i = 0
        j = 0 #Show an image during initial training (for testing purposes)

        #check directory exists:
        print('your path exists:',os.path.exists(image_directory))
        for folder in glob.iglob(image_directory+'/*'):
            print(os.path.basename(folder))
            j=0
            for sample in tqdm(glob.iglob(folder+'/*.jpg')):
                img = cv.imread(sample,cv.IMREAD_GRAYSCALE)
                #ret,img = cv.threshold(img,254,255,cv.THRESH_TOZERO)
                # kernel = np.ones((3,3),'uint8')
                # img = np.array(img)
                # img = cv.erode(img,kernel,iterations=2)
                #$reduce size to speed up models:
                ret,img = cv.threshold(img,240,255,cv.THRESH_TOZERO)
                img = cv.resize(img,(24, 24))
                et,img = cv.threshold(img,240,255,cv.THRESH_TOZERO)
                img = cv.resize(img,(12, 12))
                et,img = cv.threshold(img,240,255,cv.THRESH_TOZERO)
                

                if j == 0:
                    j =1
                    print(img.shape)
                    cv.imshow('image',img)
                    cv.waitKey(1000)


                img = 255-img #set white to zero (background)
                img = img/255 #normalize
                img = img.ravel()
                img = img.tolist()
                data.append(img)
                label.append(i)

            #creating the lables as integers
            label_dict[i] = os.path.basename(folder)
            i = i+1
        
        data = pd.DataFrame(data)
        #creating new column from the target list
        data["label"] = label
        #shuffeling the data
        data = data.sample(frac=1)
        #print(data)
        
        #Save dataframe to avoid reloading it slowly in future
        file = save_directory+'\\'+filename
        pickle.dump(data,open(file,'wb'))
        
        file = save_directory+'\\'+'dictionary'+filename
        pickle.dump(label_dict,open(file,'wb'))

        return data


    # In[76]:

    def svm_model(self,X_train, X_test, y_train, y_test,directory,data_name):
        from sklearn import svm
        svm_classifier = svm.SVC(kernel='rbf',gamma=0.001,C=5)
        svm_classifier.fit(X_train, y_train)
        predicted = svm_classifier.predict(X_test)
        accuracy = (len(X_test[predicted==y_test])/len(X_test))*100
        filename = directory+'\\'+data_name+'svm_classifier.sav'
        pickle.dump(svm_classifier,open(filename,'wb'))
        return  accuracy,svm_classifier

    def gaussian_naive_bayes(self,X_train, X_test, y_train, y_test,directory,data_name):
        from sklearn.naive_bayes import GaussianNB
        GNB_classifier = GaussianNB()
        GNB_classifier.fit(X_train, y_train)
        predicted = GNB_classifier.predict(X_test)
        accuracy = (len(X_test[predicted == y_test]) / len(X_test)) * 100
        filename = directory+'\\'+data_name+'GNB_classifier.sav'
        pickle.dump(GNB_classifier,open(filename,'wb'))
        return accuracy,GNB_classifier

    def decision_tree(self,X_train, X_test, y_train, y_test,directory,data_name):
        from sklearn import tree
        dt_classifier = tree.DecisionTreeClassifier()
        dt_classifier.fit(X_train, y_train)
        predicted = dt_classifier.predict(X_test)
        accuracy = (len(X_test[predicted == y_test]) / len(X_test)) * 100
        filename = directory+'\\'+data_name+'dt_classifier.sav'
        pickle.dump(dt_classifier,open(filename,'wb'))
        return accuracy,dt_classifier


    def k_nearest_neighbors(self,X_train, X_test, y_train, y_test,directory,data_name):
        from sklearn.neighbors import KNeighborsClassifier
        KNN_classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        KNN_classifier.fit(X_train, y_train)
        predicted = KNN_classifier.predict(X_test)
        accuracy = (len(X_test[predicted == y_test]) / len(X_test)) * 100
        filename = directory+'\\'+data_name+'KNN_classifier.sav'
        pickle.dump(KNN_classifier,open(filename,'wb'))
        return accuracy,KNN_classifier

    def stochastic_gradient_decend(self,X_train, X_test, y_train, y_test,directory,data_name):
        from sklearn.linear_model import SGDClassifier
        sgd_classifier = SGDClassifier(loss="log", penalty="l2", max_iter=1000)
        sgd_classifier.fit(X_train, y_train)
        predicted = sgd_classifier.predict(X_test)
        accuracy = (len(X_test[predicted == y_test]) / len(X_test)) * 100
        filename = directory+'\\'+data_name+'sgd_classifier.sav'
        pickle.dump(sgd_classifier,open(filename,'wb'))
        pred_all = sgd_classifier.predict_proba(X_test)
        return accuracy,sgd_classifier

    def predict(self,image,model_number,save_directory,dataset_name,show = False):

        #note - dataset_name should not include the .sav portion of the filename
        file = save_directory+'\\dictionary'+dataset_name+'.sav'
        file_dict = pickle.load(open(file,'rb'))
        try:
            img = cv.imread(image,cv.IMREAD_GRAYSCALE)
        except:
            img = image
        #reduce size to speed up models:
        
        et,img = cv.threshold(img,240,255,cv.THRESH_TOZERO)
        img = cv.resize(img,(24, 24))
        et,img = cv.threshold(img,240,255,cv.THRESH_TOZERO)
        img = cv.resize(img,(12, 12))
        et,img = cv.threshold(img,240,255,cv.THRESH_TOZERO)
        if show == True:
            cv.imshow('id_image',img)
            cv.waitKey(5000)
        img = np.array(img)
        img = 255-img #set white to zero (background)
        img = img/255 #normalize
        img = img.ravel()    
        img = img.reshape(1, -1)
        img = img.tolist()
        models = [dataset_name+'svm_classifier.sav',dataset_name+'GNB_classifier.sav',dataset_name+'dt_classifier.sav',dataset_name+'KNN_classifier.sav',dataset_name+'sgd_classifier.sav']
        filename = os.path.join(save_directory,models[model_number])
        classifier = pickle.load(open(filename,'rb'))
        predicted_Num = classifier.predict(img)
        predicted_Char = file_dict[predicted_Num[0]]
        pred_all = classifier.predict_proba(img)
        return(predicted_Num,predicted_Char,pred_all)
        
        
        


# In[78]:

if __name__ == "__main__":
    
    test = image_id(load_dataframe = True, load_models=False ,predict_only=True)
    path = os.getcwd()
    delimiter = '\\'
    folders = path.split(delimiter)
    #print(folders)
    directory_images = folders[0:-1] + ['cs7641-project','docs', 'assets', 'sample_images']
    directory_images = delimiter.join(directory_images)

    model_directory = folders[0:-1]+['cs7641-project','docs','assets','models']
    model_directory = delimiter.join(model_directory)
    
    saved_images_on_pc = ['C:','data','extracted_images']
    saved_images_on_pc = delimiter.join(saved_images_on_pc)
    
    data_name = 'full_dataset'

    dataset_name = data_name+'.sav'



    if test.predict_only == False:
        if test.load_dataset == False:
            
            print('preprocessing begin')
            print(saved_images_on_pc)
            pr = test.create_dataframe(saved_images_on_pc,model_directory,dataset_name)
            print('preprocessing complete')
            pr = pr.sample(frac=1)
            
        if test.load_dataset == True:
            file_exists = exists(model_directory+'\\'+dataset_name)
            if file_exists:
                print('start data load')
                pr = pickle.load(open(model_directory+'\\'+dataset_name,'rb'))
                print('data loaded')
            else:
                print('No data found, check save_directory and data_name. Do the files actually exist?')
                print(model_directory+'\\'+dataset_name)
        data = pr.iloc[:,:-1]
        target = pr.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(pr.iloc[:,:-1], pr.iloc[:,-1], test_size=0.3, shuffle=False)

        if test.load_models == False:
            print('training models...')

            decision_tree_accuracy,dt_classifier = test.decision_tree(X_train, X_test, y_train, y_test,model_directory,data_name)
            print("decision tree accuracy:" + str(decision_tree_accuracy))

            sgd_accuracy,sgd_classifier = test.stochastic_gradient_decend(X_train, X_test, y_train, y_test,model_directory,data_name)
            print("stochastic gradient decent accuracy:" + str(sgd_accuracy))

            knn_accuracy,knn_classifier = test.k_nearest_neighbors(X_train, X_test, y_train, y_test,model_directory,data_name)
            print("knn accuracy:"+ str(knn_accuracy))

            svm_model_accuracy,svm_classifier = test.svm_model(X_train, X_test, y_train, y_test,model_directory,data_name)
            print("svm accuracy:" + str(svm_model_accuracy))

            gaussian_naive_bayes_accuracy,GNB_classifier = test.gaussian_naive_bayes(X_train, X_test, y_train, y_test,model_directory,data_name)
            print("gaussian naive bayes accuracy:" + str(gaussian_naive_bayes_accuracy))








    # In[79]:


    #imagedir = saved_images_on_pc+'//2//2_11232.jpg'
    imagedir = 'docs\\assets\\images\\Set_2\\Equation29.png'

    testA = test.predict(imagedir,2,model_directory,data_name,show = True)
    print(testA[1])





# %%
