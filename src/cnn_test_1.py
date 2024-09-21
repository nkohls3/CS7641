# Ask Sebastian if you have any questions.
# See the teams drive the following necessary files (2.8 Gb each):
# dictionarytest1.sav
# NN1.sav
# test1.sav

# To test a single image with this code, create an object "image_id", then image_id.single_image(img) where img is
# the string location of a 45x45 pixel image, without padding, of your symbol.
# The output of above then goes through the object

# You'll also want the environment yml file - environment.yml in the same location.

# Model was trained twice - first 10 epochs with LR = 0.003, the last 10 with LR = 0.0003

#To just use this function to predict outputs, use the "predict_symbols" class. Inputs are an array to locate the neural net and an array to locate the dictionary
# then use predict_symbols.predict(img_array), where img_array is an array of numpy arrays (of images).

#from cv2 import IMREAD_GRAYSCALE, INTER_AREA
import numpy as np
import os
import pandas as pd
import glob
import pickle
import cv2 as cv
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import train_test_split
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

from pytorch_lightning.core.decorators import auto_move_data


class ImageDataset(Dataset):
    def __init__(self, samples, labels):

        # Convert Dataframes to Numpy Arrays
        self.samples = samples.to_numpy()
        self.labels = labels.to_numpy()

    def __getitem__(self, index):
        #res = self.samples[index].shape[0]
        #res = int(np.sqrt(res))
        return torch.tensor(self.samples[index]).reshape(1, 30, 30).float(), self.labels[index]

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
                    #res = np.random.randint(20,30)
                    img = cv.resize(img, (30, 30), cv.INTER_AREA)
                    img = img/255  # normalize
                    #img = np.reshape(img,(1,res,res))
                    #output = np.full((30,30),0,dtype = np.uint8)
                    #x_center = (31 - res) // 2
                    #y_center = (31 - res) // 2
                    #output[y_center:y_center+res, x_center:x_center+res] = img
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

    def single_image(img,filename = True):
        new_image_size = [55, 55]
        border_color = 255
        if filename == True:
            img = cv.imread(img, cv.IMREAD_GRAYSCALE)
        old_image_height, old_image_width = img.shape
        new_image_width, new_image_height = new_image_size
        output = np.full((new_image_height, new_image_width),
                         border_color, dtype=np.uint8)
        # compute center offset
        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2

        # copy img image into center of result image:
        output[y_center:y_center+old_image_height,
               x_center:x_center+old_image_width] = img

        # Threshold image:
        ret, thresh1 = cv.threshold(output, 50, 255, cv.THRESH_BINARY)
        img = 255-thresh1
        #img = cv.morphologyEx(img,cv.MORPH_CLOSE,cv.getStructuringElement(cv.MORPH_RECT,(3,3)))
        #img = cv.dilate(img,cv.getStructuringElement(cv.MORPH_RECT,(3,3)),iterations = 1)
        img = cv.resize(img, (30, 30), cv.INTER_AREA)
        return img


class ResNetMNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = resnet18(num_classes=82)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.loss = nn.CrossEntropyLoss()

    def dataset(self, train_dataset, test_dataset):
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
        return torch.optim.Adam(self.parameters(), lr=0.000001)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, 256, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_ds, 256, shuffle=False)

    def validation_step(self, batch, batch_no):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        score = torch.tensor(accuracy_score(y.cpu(), torch.argmax(
            torch.softmax(logits, dim=1), dim=1).detach().cpu()))

        return {'val_loss': loss, "acc_score": score}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['acc_score'] for x in outputs]).mean()
        log = {'avg_val_loss': val_loss, "avg_val_acc": val_acc}
        print("\n\nValidation Loss: ", val_loss)
        print("\nValidation Accuracy: ", val_acc)
        return {'val_loss': val_loss, 'log': log}

    def predict(self, img, dictionary):
        img = img/255.
        logits = self(torch.tensor(img.reshape((1, 30, 30))).unsqueeze(0))
        output = torch.argmax(torch.softmax(
            logits, dim=1), dim=1).detach().cpu()
        file = open(dictionary, 'rb')
        dictionary = pickle.load(file)
        file.close()
        letter = dictionary[output.item()]
        return output, letter

class predict_symbols:
    def __init__(self,model_source,dictionary_source):
        self.model = ResNetMNIST()
        self.model.model = torch.load(delimiter.join(
            model_source))
        self.model = self.model.eval()
        self.dictionary_source = dictionary_source
        
    def predict(self,images_array):
        output = []
        number = []
        for image in range(len(images_array)):
            img = image_id.single_image(images_array[image],filename = False)
            num, out = self.model.predict(img, delimiter.join(self.dictionary_source))
            output.append(out)
            number.append(num)
        #print(summary(self.model,(1,30,30)))

        return output,number

class results_analysis:
    def __init__(self,y_test_target,y_test_real,dictionary):
        delimiter = '//'
        self.y_pred = y_test_target
        self.y = y_test_real
        file = open(delimiter.join(dictionary),'rb')
        self.dictionary = pickle.load(file)
        file.close()
    def roc_score(self):

        letter = 0
        for j in range(81):
            letter = j
            y_pred = self.y_pred.detach().numpy()
            y = self.y[0:20000].to_numpy()
            z = np.zeros((20000,1))
            zz = np.zeros((20000,1))
            for i in range(20000):
                if y[i] == letter:
                    z[i] = 1
                else:
                    z[i] = 0
                zz[i] = y_pred[i,letter]

            y = z
            ypred = zz
            fpr,tpr,thresholds = roc_curve(y,ypred)
            roc_auc = auc(fpr,tpr)

            display = RocCurveDisplay(fpr=fpr,tpr=tpr,roc_auc = roc_auc)
            display.plot()
            plt.title(self.dictionary[j])
            plt.show()
            print(j)



if __name__ == "__main__":

    # modifiable:
    delimiter = '\\'
    image_location = ['C:', 'data', 'modified_images']
    preload_dataset_save = ['C:', 'data', 'preload_data']
    preload_data_filename = 'test1.sav'
    load_model = True
    predict_only = True
    load_dataset = True
    analysis = False
    image_directory = delimiter.join(['C:', 'Users', 'sebastian','Documents', 'GitHub','cs7641-project', 'docs', 'assets', 'test_equations'])
    #folder = delimiter.join(['C:', 'Users', 'sebastian','Documents', 'GitHub','cs7641-project', 'docs', 'assets', 'test_equations', 'Eq2_1'])




   # Create Strings for data / save locations
    image_location = delimiter.join(image_location)
    preload_dataset_save = delimiter.join(preload_dataset_save)

    # Create Datasets

    if predict_only == False:
        dataset = image_id(load_dataset=load_dataset)
        dataset.create_dataframe(
            image_location, preload_dataset_save, preload_data_filename)
        dataset.split_data()
        model = ResNetMNIST()
        if load_model == True:
            model.model = torch.load(delimiter.join(
             ['C:', 'data', 'preload_data', 'NN1.sav']))
            

    # Create CNN:

    if load_model == True:
        if predict_only == True:
            img = []
            for folder in glob.iglob(image_directory+'/*'):
                img = []
                for sample in tqdm(glob.iglob(folder+'/*.PNG')):
                    im = cv.imread(sample, cv.IMREAD_GRAYSCALE)
                    #img = image_id.single_image(sample)
                    #cv.imshow('test',img)
                    #img = torch.tensor(img).unsqueeze(0)            
                    #cv.waitKey(5000)
                    img.append(im)
                    #output, letter = model.predict(img, delimiter.join(['C:', 'data', 'preload_data', 'dictionarytest1.sav']))
                    #print(output, letter)
                predictor = predict_symbols(['C:', 'data', 'preload_data', 'NN1.sav'],['C:', 'data', 'preload_data', 'dictionarytest1.sav'])
                print(predictor)
                    # model = ResNetMNIST()
                # model.model = torch.load(delimiter.join(
                #     ['C:', 'data', 'preload_data', 'NN1.sav']))
                if predict_only == True:
                    letters,numbers = predictor.predict(img)
                    print(letters)
            

                #model = model.eval()
                # for sample in tqdm(glob.iglob(folder+'/*.PNG')):
                #     #img = cv.imread(sample, cv.IMREAD_GRAYSCALE)
                #     img = image_id.single_image(sample)
                #     #cv.imshow('test',img)
                #     #img = torch.tensor(img).unsqueeze(0)            
                #     #cv.waitKey(5000)
                #     print(sample)
                #     output, letter = model.predict(img, delimiter.join(['C:', 'data', 'preload_data', 'dictionarytest1.sav']))
                #     print(output, letter)


    if predict_only == False:
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=2,
            progress_bar_refresh_rate=1 )
        model.dataset(dataset.train_dataset, dataset.test_dataset)
        trainer.fit(model)
        torch.save(model.model, delimiter.join(
            ['C:', 'data', 'preload_data', 'NN1.sav']))

    if analysis == True:
        dataset = image_id(load_dataset=load_dataset)
        dataset.create_dataframe(
            image_location, preload_dataset_save, preload_data_filename)
        dataset.split_data()
        model = ResNetMNIST()
        model.model = torch.load(delimiter.join(
             ['C:', 'data', 'preload_data', 'NN1.sav']))
        model.eval
        inputdata = dataset.X_test.iloc[0:20000,:].to_numpy()
        inputdata = inputdata.reshape((20000,30,30)).tolist()
        res = results_analysis(torch.softmax(model(torch.tensor(inputdata).unsqueeze(1)),dim=1),dataset.y_test,['C:', 'data', 'preload_data', 'dictionarytest1.sav'])

        res.roc_score()
