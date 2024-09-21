#Ask Sebastian if you have any questions.
#See the teams drive the following necessary files (2.8 Gb each):
##dictionarytest1.sav
##NN1.sav
##test1.sav

#To test a single image with this code, create an object "image_id", then image_id.single_image(img) where img is
# the string location of a 45x45 pixel image, without padding, of your symbol.
# The output of above then goes through the object 

#You'll also want the environment yml file - environment.yml in the same location.

##Model was trained twice - first 10 epochs with LR = 0.003, the last 10 with LR = 0.0003

import numpy as np
import os
import pandas as pd
import glob
import pickle

from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from os.path import exists
from sklearn.model_selection import train_test_split
from sklearn import metrics
import cv2 as cv
import torch
import torchvision
import numpy as np
import matplotlib as plt
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data


class ImageDataset(Dataset):
    def __init__(self, samples, labels):

        # Convert Dataframes to Numpy Arrays
        self.samples = samples.to_numpy()
        self.labels = labels.to_numpy()

    def __getitem__(self, index):
        return torch.tensor(self.samples[index]).reshape(1, 28, 28).float(), self.labels[index]

    def __len__(self):
        return len(self.samples)


class image_id:
    def __init__(self, load_dataset=True):
        self.load_dataset = load_dataset
        self.delimiter = '\\'

    def create_dataframe(self, image_directory, save_directory, filename):

        if self.load_dataset == True:
            file = self.delimiter.join([save_directory, filename])
            data = pickle.load(open(file, 'rb'))
            file = self.delimiter.join([save_directory, 'dictionary'+filename])
            file_dict = pickle.load(open(file, 'rb'))
        else:
            data = []
            label = []
            label_dict = {}
            i = 0
            j = 0

            # check directory exists:
            print('your path exists:', os.path.exists(image_directory))
            for folder in glob.iglob(image_directory+'/*'):
                print(os.path.basename(folder))
                j = 0
                for sample in tqdm(glob.iglob(folder+'/*.jpg')):
                    img = cv.imread(sample, cv.IMREAD_GRAYSCALE)

                    img = cv.resize(img, (28, 28), cv.INTER_AREA)

                    img = img/255  # normalize
                    img = img.ravel()
                    img = img.tolist()
                    data.append(img)
                    label.append(i)

                # creating the lables as integers
                label_dict[i] = os.path.basename(folder)
                i = i+1

            data = pd.DataFrame(data)
            # creating new column from the target list
            data["label"] = label
            # shuffeling the data
            data = data.sample(frac=1)
            # print(data)

            # Save dataframe to avoid reloading it slowly in future
            file = save_directory+'\\'+filename
            pickle.dump(data, open(file, 'wb'))

            file = save_directory+'\\'+'dictionary'+filename
            pickle.dump(label_dict, open(file, 'wb'))

        self.data = data

    def split_data(self):
        pr = self.data.sample(frac=1, random_state=42)
        prdata = pr.iloc[:, :-1]
        prtarget = pr.iloc[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            prdata, prtarget, test_size=0.3, shuffle=False)
        self.train_dataset = ImageDataset(self.X_train, self.y_train)
        self.test_dataset = ImageDataset(self.X_test, self.y_test)
        print(np.max(self.y_train.to_numpy()))
        print(np.max(self.y_test.to_numpy()))
    def single_image(img):
        new_image_size = [65,65]
        border_color = 255
        img = cv.imread(img,cv.IMREAD_GRAYSCALE)
        old_image_height, old_image_width = img.shape
        new_image_width, new_image_height = new_image_size
        color = 255
        output = np.full((new_image_height,new_image_width),border_color,dtype = np.uint8)
        ## compute center offset
        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2

        #copy img image into center of result image:
        output[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = img

        #Threshold image:
        ret,thresh1 = cv.threshold(output,50,255,cv.THRESH_BINARY)
        img = 255-thresh1
        return img


class ResNetMNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = resnet18(num_classes=82)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.loss = nn.CrossEntropyLoss()

    def dataset(self,train_dataset,test_dataset):
        self.train_ds = train_dataset
        self.test_ds = test_dataset

    @auto_move_data
    def forward(self, x):
        return self.model(x.float())

    def training_step(self, batch, batch_no):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.003)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, 256, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_ds, 256, shuffle=False)

    def validation_step(self, batch, batch_no):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        
        score = torch.tensor(accuracy_score(y.cpu(),torch.argmax(torch.softmax(logits,dim=1),dim=1).detach().cpu()))

        return {'val_loss': loss, "acc_score": score}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['acc_score'] for x in outputs]).mean()
        log = {'avg_val_loss': val_loss, "avg_val_acc": val_acc}
        print("\n\nValidation Loss: ", val_loss)
        print("\nValidation Accuracy: ", val_acc)
        return {'val_loss': val_loss, 'log': log}

    def predict(self,img,dictionary):
        logits = self(img)
        output = torch.argmax(torch.softmax(logits,dim=1),dim=1).detach().cpu()
        letter = dictionary[output]
        return output,letter


if __name__ == "__main__":

    #modifiable: 
    delimiter = '\\'
    image_location = ['C:', 'data', 'modified_images']
    preload_dataset_save = ['C:', 'data', 'preload_data']
    preload_data_filename = 'test1.sav'
    load_model = True
    predict_only = False
    img = None #Put filename, location here to image to sample.


    #Create Strings for data / save locations
    image_location = delimiter.join(image_location)
    preload_dataset_save = delimiter.join(preload_dataset_save)

    #Create Datasets
    dataset = image_id(load_dataset=True)
    dataset.create_dataframe(
        image_location, preload_dataset_save, preload_data_filename)
    dataset.split_data()

    #Create CNN:
    

    if load_model == True:
        model = ResNetMNIST()
        model.model = torch.load(delimiter.join(['C:', 'data', 'preload_data','NN1.sav']))
        if predict_only == True:
            img = image_id.single_image(img)
            model.predict(img)
    else:
        model = ResNetMNIST()
    


    if predict_only == False:
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=10,
            progress_bar_refresh_rate=1
        )
        model.dataset(dataset.train_dataset,dataset.test_dataset)
        trainer.fit(model)
        torch.save(model.model,delimiter.join(['C:', 'data', 'preload_data','NN1.sav']))



    

    
