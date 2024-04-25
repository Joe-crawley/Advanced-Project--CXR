import time

import PIL.Image
import sklearn.metrics
import torch
import torch.nn as nn
import torch.optim as optim
#from torchcam.methods import GradCAM
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc,roc_auc_score,classification_report,precision_recall_curve
from sklearn.model_selection import train_test_split

import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
from transformers import ViTImageProcessor,ViTForImageClassification,AutoImageProcessor, SwinModel,ViTFeatureExtractor,Swinv2Model,Swinv2ForImageClassification
from torch.utils.data import Dataset,DataLoader,Subset

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
from datetime import datetime
import matplotlib
import os
from PIL import Image

import timm

# print(torch.cuda.is_available())
# cudnn.benchmark = True
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# torch.cuda.is_available()
# torch.cuda.device_count()
# torch.cuda.current_device()
#
# print(torch.cuda.get_device_name(0))
classEncoding = {
        'Atelectasis': torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Consolidation': torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Infiltration': torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Pneumothorax': torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Edema': torch.FloatTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Emphysema': torch.FloatTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Fibrosis': torch.FloatTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Effusion': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        'Pneumonia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        'Pleural_Thickening': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        'Cardiomegaly': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        'Nodule': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        'Hernia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
        'Mass': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
        'No Finding': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    }
def ClassImbalanceHelper():
    df = pd.read_csv('Data/Data_Entry_2017.csv')
    test = df['Finding Labels'].str.split('|').str[0]
    CountDict = test.value_counts()

    InverseProbDict = 1 / (CountDict.div(sum(CountDict)))

    return dict(InverseProbDict)

class ClassAveragedBCELoss(nn.Module):
    def __init__(self):
        super(ClassAveragedBCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # Use 'none' to get the loss per element

    def forward(self, logits, targets):
        # Compute BCE loss without reduction
        per_sample_losses = self.bce_loss(logits, targets)

        # Average across classes
        per_class_average_loss = per_sample_losses.mean(dim=0)

        # Average across the batch (or further average across classes if required)
        return per_class_average_loss.mean()
class DataSetHelper(Dataset):
    classEncoding = {
        'Atelectasis': torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Consolidation': torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Infiltration': torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Pneumothorax': torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Edema': torch.FloatTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Emphysema': torch.FloatTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Fibrosis': torch.FloatTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Effusion': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        'Pneumonia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        'Pleural_Thickening': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        'Cardiomegaly': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        'Nodule': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        'Hernia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
        'Mass': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
        'No Finding': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    }


    def __init__(self, transforms, classEncoding=classEncoding):
        self.image_names = []
        self.labels = []
        self.transforms = transforms
        file = open('Data/Data_Entry_2017.csv', 'r')
        start = True
        for line in file:
            if (start):
                start = False
                continue
            items = str(line).split(',')
            image_name = items[0]
            image_name = os.path.join('Data/images_001/images', image_name)
            self.image_names.append(image_name)
            label = items[1]
            labelDisease = label.split('|')
            labelTensor = classEncoding[labelDisease[0]]
            self.labels.append(labelTensor)
        # self.image_names = np.array(self.image_names)
        # self.labels = np.array(self.labels)
    def __getitem__(self, index):
        image_path = self.image_names[index]
        image = Image.open(image_path, 'r').convert('RGB')
        image = self.transforms(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.labels)
class DataSetHelperMultiLabel(Dataset):



    def __init__(self, transforms, classEncoding=classEncoding):
        MLclassEncoding = {
            'Atelectasis': torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'Consolidation': torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'Infiltration': torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'Pneumothorax': torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'Edema': torch.FloatTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'Emphysema': torch.FloatTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            'Fibrosis': torch.FloatTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
            'Effusion': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
            'Pneumonia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
            'Pleural_Thickening': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
            'Cardiomegaly': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
            'Nodule': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
            'Hernia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
            'Mass': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
            # 'No Finding': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        }
        self.image_names = []
        self.labels = []
        self.transforms = transforms
        file = open('Data/Data_Entry_2017.csv', 'r')
        start = True
        for line in file:
            if (start):
                start = False
                continue
            items = str(line).split(',')
            image_name = items[0]
            image_name = os.path.join('Data/images_001/images', image_name)
            self.image_names.append(image_name)
            label = items[1]
            labelDisease = label.split('|')
            ###MULTI LABEL MAGIC
            labelTensor = torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            for disease in labelDisease:
                if disease != 'No Finding':

                    labelTensor += MLclassEncoding[disease]




            self.labels.append(labelTensor)
        # self.image_names = np.array(self.image_names)
        # self.labels = np.array(self.labels)
    def __getitem__(self, index):
        image_path = self.image_names[index]
        image = Image.open(image_path, 'r').convert('RGB')
        image = self.transforms(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.labels)
class DataSetHelperMini(Dataset):
    classEncoding = {
        'Atelectasis': torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Consolidation': torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Infiltration': torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Pneumothorax': torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Edema': torch.FloatTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Emphysema': torch.FloatTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Fibrosis': torch.FloatTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Effusion': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        'Pneumonia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        'Pleural_Thickening': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        'Cardiomegaly': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        'Nodule': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        'Hernia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
        'Mass': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
        'No Finding': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    }
    # classEncoding = {
    #     'Atelectasis': 0,
    #     'Consolidation': 1,
    #     'Infiltration': 2,
    #     'Pneumothorax': 3,
    #     'Edema': 4,
    #     'Emphysema': 5,
    #     'Fibrosis': 6,
    #     'Effusion': 7,
    #     'Pneumonia': 8,
    #     'Pleural_Thickening': 9,
    #     'Cardiomegaly': 10,
    #     'Nodule': 11,
    #     'Hernia': 12,
    #     'Mass': 13,
    #     'No Finding': 14,
    #
    # }

    def __init__(self, transforms,Length, classEncoding=classEncoding):
        self.image_names = []
        self.labels = []
        self.transforms = transforms
        file = open('Data/Data_Entry_2017.csv', 'r')
        start = True
        i = 0
        for line in file:
            if (start):
                start = False
                continue
            if i > Length:
                pass
            else:

                items = str(line).split(',')
                image_name = items[0]
                image_name = os.path.join('Data/images_001/images', image_name)
                self.image_names.append(image_name)
                label = items[1]
                labelDisease = label.split('|')
                labelTensor = classEncoding[labelDisease[0]]
                self.labels.append(labelTensor)
                i += 1

    def __getitem__(self, index):
        image_path = self.image_names[index]
        image = Image.open(image_path, 'r').convert('RGB')
        image = self.transforms(image)
        return image, ((self.labels[index]))
    def __len__(self):
        return len(self.labels)
    def SwitchTransforms(self,Transforms):   #SET
        self.transforms = Transforms

class TrainSplit(DataSetHelper):
    classEncoding = {
        'Atelectasis': torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Consolidation': torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Infiltration': torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Pneumothorax': torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Edema': torch.FloatTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Emphysema': torch.FloatTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Fibrosis': torch.FloatTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Effusion': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        'Pneumonia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        'Pleural_Thickening': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        'Cardiomegaly': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        'Nodule': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        'Hernia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
        'Mass': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
        'No Finding': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    }
    # classEncoding = {
    #     'Atelectasis': 0,
    #     'Consolidation': 1,
    #     'Infiltration': 2,
    #     'Pneumothorax': 3,
    #     'Edema': 4,
    #     'Emphysema': 5,
    #     'Fibrosis': 6,
    #     'Effusion': 7,
    #     'Pneumonia': 8,
    #     'Pleural_Thickening': 9,
    #     'Cardiomegaly': 10,
    #     'Nodule': 11,
    #     'Hernia': 12,
    #     'Mass': 13,
    #     'No Finding': 14,
    #
    # }

    def __init__(self, transforms, classEncoding=classEncoding):
        DataSetHelper.__init__(self, transforms, classEncoding=classEncoding)
        trainFile = open('Data/train_val_list.txt', 'r')
        self.trainingFiles = []
        self.trainingLabels = []
        for line in trainFile:
            line = line.strip()
            filename = os.path.join('Data/images_001/images', line)
            self.trainingFiles.append(filename)
            self.trainingLabels.append(self.labels[self.image_names.index(filename)])



    def __getitem__(self, index):
        #DataIndex = self.image_names.index(self.trainingFiles[index])

        fileName = self.trainingFiles[index]
        label = self.trainingLabels[index]
        image = Image.open(fileName, 'r').convert('RGB')
        image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.trainingFiles)
class TrainSplitMultiLabel(DataSetHelperMultiLabel):
    classEncoding = {
        'Atelectasis': torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Consolidation': torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Infiltration': torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Pneumothorax': torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Edema': torch.FloatTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Emphysema': torch.FloatTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Fibrosis': torch.FloatTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Effusion': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        'Pneumonia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        'Pleural_Thickening': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        'Cardiomegaly': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        'Nodule': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        'Hernia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
        'Mass': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
        'No Finding': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    }
    # classEncoding = {
    #     'Atelectasis': 0,
    #     'Consolidation': 1,
    #     'Infiltration': 2,
    #     'Pneumothorax': 3,
    #     'Edema': 4,
    #     'Emphysema': 5,
    #     'Fibrosis': 6,
    #     'Effusion': 7,
    #     'Pneumonia': 8,
    #     'Pleural_Thickening': 9,
    #     'Cardiomegaly': 10,
    #     'Nodule': 11,
    #     'Hernia': 12,
    #     'Mass': 13,
    #     'No Finding': 14,
    #
    # }

    def __init__(self, transforms, classEncoding=classEncoding):
        DataSetHelperMultiLabel.__init__(self, transforms, classEncoding=classEncoding)
        trainFile = open('Data/train_val_list.txt', 'r')
        self.trainingFiles = []
        self.trainingLabels = []
        for line in trainFile:
            line = line.strip()
            filename = os.path.join('Data/images_001/images', line)
            self.trainingFiles.append(filename)
            self.trainingLabels
            self.trainingLabels.append(self.labels[self.image_names.index(filename)])



    def __getitem__(self, index):
        #DataIndex = self.image_names.index(self.trainingFiles[index])

        fileName = self.trainingFiles[index]
        label = self.trainingLabels[index]
        image = Image.open(fileName, 'r').convert('RGB')
        image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.trainingFiles)

class CustomDataset(Dataset):
    def __init__(self,data,labels,transforms=None):
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image,label = self.data[idx],self.labels[idx]
        if self.transforms:
            image = self.transforms(image)
        return image,label
class TestSplit(DataSetHelper):
    classEncoding = {
        'Atelectasis': torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Consolidation': torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Infiltration': torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Pneumothorax': torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Edema': torch.FloatTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Emphysema': torch.FloatTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Fibrosis': torch.FloatTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Effusion': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        'Pneumonia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        'Pleural_Thickening': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        'Cardiomegaly': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        'Nodule': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        'Hernia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
        'Mass': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
        'No Finding': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    }
    ClassWeights = ClassImbalanceHelper()
    # classEncoding = {
    #     'Atelectasis': 0,
    #     'Consolidation': 1,
    #     'Infiltration': 2,
    #     'Pneumothorax': 3,
    #     'Edema': 4,
    #     'Emphysema': 5,
    #     'Fibrosis': 6,
    #     'Effusion': 7,
    #     'Pneumonia': 8,
    #     'Pleural_Thickening': 9,
    #     'Cardiomegaly': 10,
    #     'Nodule': 11,
    #     'Hernia': 12,
    #     'Mass': 13,
    #     'No Finding': 14,
    #
    # }

    def __init__(self, transforms, classEncoding=classEncoding):
        DataSetHelper.__init__(self, transforms, classEncoding=classEncoding)
        trainFile = open('Data/test_list.txt', 'r')
        self.trainingFiles = []
        self.trainingLabels = []
        for line in trainFile:
            line = line.strip()
            filename = os.path.join('Data/images_001/images', line)
            self.trainingFiles.append(filename)
            self.trainingLabels.append(self.labels[self.image_names.index(filename)])

    def __getitem__(self, index):
        #DataIndex = self.image_names.index(self.trainingFiles[index])

        fileName = self.trainingFiles[index]
        label = self.trainingLabels[index]
        image = Image.open(fileName, 'r').convert('RGB')
        image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.trainingFiles)
class TestSplitMultiLabel(DataSetHelperMultiLabel):
    classEncoding = {
        'Atelectasis': torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Consolidation': torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Infiltration': torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Pneumothorax': torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Edema': torch.FloatTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Emphysema': torch.FloatTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Fibrosis': torch.FloatTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        'Effusion': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        'Pneumonia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        'Pleural_Thickening': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        'Cardiomegaly': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        'Nodule': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
        'Hernia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
        'Mass': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        #'No Finding': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    }
    ClassWeights = ClassImbalanceHelper()
    # classEncoding = {
    #     'Atelectasis': 0,
    #     'Consolidation': 1,
    #     'Infiltration': 2,
    #     'Pneumothorax': 3,
    #     'Edema': 4,
    #     'Emphysema': 5,
    #     'Fibrosis': 6,
    #     'Effusion': 7,
    #     'Pneumonia': 8,
    #     'Pleural_Thickening': 9,
    #     'Cardiomegaly': 10,
    #     'Nodule': 11,
    #     'Hernia': 12,
    #     'Mass': 13,
    #     'No Finding': 14,
    #
    # }

    def __init__(self, transforms, classEncoding=classEncoding):
        DataSetHelperMultiLabel.__init__(self, transforms, classEncoding=classEncoding)
        trainFile = open('Data/test_list.txt', 'r')
        self.trainingFiles = []
        self.trainingLabels = []
        for line in trainFile:
            line = line.strip()
            filename = os.path.join('Data/images_001/images', line)
            self.trainingFiles.append(filename)
            self.trainingLabels.append(self.labels[self.image_names.index(filename)])

    def __getitem__(self, index):
        #DataIndex = self.image_names.index(self.trainingFiles[index])

        fileName = self.trainingFiles[index]
        label = self.trainingLabels[index]
        image = Image.open(fileName, 'r').convert('RGB')
        image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.trainingFiles)

class ModelProcess():

    def __init__(self, model, in_transforms, device,flag,endToEnd = True,multiLabel = False):

        #self.NumClasses = 15
        self.multiLabel = multiLabel
        if self.multiLabel:
            self.NumClasses = 14
        else:
            self.NumClasses = 15
        self.model = model
        self.device = device
        self.endToEnd = endToEnd
        self.flag = flag
        if flag =='MbNv2':
            MobileNetV2Transforms = transforms.Compose([transforms.Resize((224, 224)),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.RandomRotation(degrees=15),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])

                                                        ]
                                                       )

            self.transforms = MobileNetV2Transforms

            self.model = models.mobilenet_v2(pretrained=True).to(device)
            if not endToEnd:

                for param in self.model.parameters():
                    param.requires_grad = False
            self.model.classifier[1] = nn.Linear(self.model.last_channel,self.NumClasses).to(device)

            ##ADd activation function !!
            for param in self.model.classifier[1].parameters():
                param.requires_grad = True

        elif flag == 'ENet-b0':
            EfficientNetTransforms = transforms.Compose([transforms.Resize((224, 224)),
                                                         transforms.RandomHorizontalFlip(),
                                                         transforms.RandomRotation(degrees=15),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225])

                                                         ]
                                                        )
            model_name = 'efficientnet_b0'
            self.model = timm.create_model(model_name, pretrained=True).to(device)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, self.NumClasses).to(device)
            self.transforms = EfficientNetTransforms
            for param in self.model.parameters():
                param.requires_grad = True

        elif flag == 'ENet-b1':
            EfficientNetTransforms = transforms.Compose([transforms.Resize((224, 224)),
                                                        transforms.RandomHorizontalFlip(),
                                                         transforms.RandomRotation(degrees=15),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])

                                                        ]
                                                       )
            model_name = 'efficientnet_b1'
            self.model = timm.create_model(model_name,pretrained=True).to(device)
            self.model.classifier = nn.Linear(self.model.classifier.in_features,self.NumClasses).to(device)
            self.transforms = EfficientNetTransforms
            for param in self.model.parameters():
                param.requires_grad = True

        elif flag == 'ENet-b4':
            EfficientNetTransforms = transforms.Compose([transforms.Resize((224, 224)),
                                                         transforms.RandomHorizontalFlip(),
                                                         transforms.RandomRotation(degrees=15),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225])

                                                         ]
                                                        )
            model_name = 'efficientnet_b4'
            self.model = timm.create_model(model_name, pretrained=True).to(device)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, self.NumClasses).to(device)
            self.transforms = EfficientNetTransforms
            for param in self.model.parameters():
                param.requires_grad = True

        elif flag == 'ENet-b5':
            EfficientNetTransforms = transforms.Compose([transforms.Resize((224, 224)),
                                                         transforms.RandomHorizontalFlip(),
                                                         transforms.RandomRotation(degrees=15),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225])

                                                         ]
                                                        )
            model_name = 'efficientnet_b5'
            self.model = timm.create_model(model_name, pretrained=True).to(device)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, self.NumClasses).to(device)
            self.transforms = EfficientNetTransforms
            for param in self.model.parameters():
                param.requires_grad = True

        elif flag == 'Swin-Small':
            TempTransforms = transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(degrees=15),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
            model_name = 'swin_small_patch4_window7_224.ms_in22k'
            self.model = timm.create_model(model_name,pretrained=True,num_classes=self.NumClasses).to(device)
            self.transforms = TempTransforms
        elif flag == 'Swin-Base':
            TempTransforms = transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(degrees=15),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
            model_name = 'swin_base_patch4_window7_224.ms_in22k'

            self.model = timm.create_model(model_name,pretrained=True,num_classes=self.NumClasses).to(device)
            self.transforms = TempTransforms
            for param in self.model.parameters():
                param.requires_grad = True
        elif flag == 'Swin-Large':
            TempTransforms = transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(degrees=15),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
            model_name = 'swin_large_patch4_window7_224.ms_in22k'
            self.model = timm.create_model(model_name,pretrained=True,num_classes=self.NumClasses).to(device)
            self.transforms = TempTransforms
            for param in self.model.parameters():
                param.requires_grad = True
        elif flag == 'Swin-V2-Base':
            TempTransforms = transforms.Compose([transforms.Resize((256, 256)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(degrees=15),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
            model_name = 'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k'
            model_name = 'hf_hub:timm/swinv2_base_window12to16_192to256.ms_in22k_ft_in1k'
            #model_name = 'hf_hub:timm/swinv2_base_window12to24_192to256.ms_in22k_ft_in1k'
            self.model = timm.create_model(model_name, pretrained=True, num_classes=self.NumClasses).to(device)
            self.transforms = TempTransforms
            for param in self.model.parameters():
                param.requires_grad = True
        elif flag == 'Swin-V2-Large':
            TempTransforms = transforms.Compose([transforms.Resize((256, 256)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(degrees=15),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
            model_name = 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k'
            self.model = timm.create_model(model_name,pretrained=True,num_classes=self.NumClasses).to(device)
            self.transforms = TempTransforms
            for param in self.model.parameters():
                param.requires_grad = True
        elif flag == 'DenseNet-121':
            TempTransforms = transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(degrees=15),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
            model_name = 'densenet121.tv_in1k'
            self.model = timm.create_model(model_name, pretrained=True, num_classes=self.NumClasses).to(device)
            self.transforms = TempTransforms
            for param in self.model.parameters():
                param.requires_grad = True
        elif flag == 'DenseNet-169':
            TempTransforms = transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(degrees=15),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
            model_name = 'densenet169.tv_in1k'
            self.model = timm.create_model(model_name, pretrained=True, num_classes=self.NumClasses).to(device)
            self.transforms = TempTransforms
            for param in self.model.parameters():
                param.requires_grad = True
        elif flag == 'DenseNet-201':
            TempTransforms = transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(degrees=15),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
            model_name = 'densenet201.tv_in1k'
            self.model = timm.create_model(model_name, pretrained=True, num_classes=self.NumClasses).to(device)
            self.transforms = TempTransforms
            for param in self.model.parameters():
                param.requires_grad = True
        elif flag == 'ENetV2-M':

            TempTransforms = transforms.Compose([transforms.Resize((480,480)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(degrees=15),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
            self.transforms = TempTransforms
            self.model = models.efficientnet_v2_m(pretrained=True).to(device)
            self.model.classifier[1] = nn.Linear(1280,self.NumClasses).to(device)
            for param in self.model.parameters():
                param.requires_grad = True

        elif flag == 'ENetV2-S':

            TempTransforms = transforms.Compose([transforms.Resize((384,384)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(degrees=15),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
            self.transforms = TempTransforms
            self.model = models.efficientnet_v2_s(pretrained=True).to(device)
            self.model.classifier[1] = nn.Linear(1280,self.NumClasses).to(device)
            for param in self.model.parameters():
                param.requires_grad = True

        elif flag == 'ENetV2-L':
            TempTransforms = transforms.Compose([transforms.Resize((480,480)),
                                                transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(degrees=15),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
            self.transforms = TempTransforms
            self.model = models.efficientnet_v2_l(pretrained=True).to(device)
            self.model.classifier[1] = nn.Linear(1280,self.NumClasses).to(device)
            for param in self.model.parameters():
                param.requires_grad = True


        elif flag == 'ViT-Small':   #BROKEN FIX !
            TempTransforms = transforms.Compose([transforms.Resize((224,224)),
                                                 transforms.RandomHorizontalFlip(),

                                                 transforms.RandomRotation(degrees=15),
                                                 #transforms.RandomResizedCrop()
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                                      std=[0.26862954, 0.26130258, 0.27577711])

                                                 ])
            self.transforms = TempTransforms
            model_name = 'vit_small_patch16_224.augreg_in21k'

            self.model = timm.create_model(model_name,pretrained=True,num_classes=self.NumClasses)
            self.model.to(device)
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True
        elif flag == 'ViT-Base':   #BROKEN FIX !
            TempTransforms = transforms.Compose([transforms.Resize((224,224)),
                                                 transforms.RandomHorizontalFlip(),

                                                 transforms.RandomRotation(degrees=15),
                                                 #transforms.RandomResizedCrop()
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                                      std=[0.26862954, 0.26130258, 0.27577711])

                                                 ])
            self.transforms = TempTransforms
            model_name = 'vit_base_patch16_224.augreg_in21k'

            self.model = timm.create_model(model_name,pretrained=True,num_classes=self.NumClasses)
            self.model.to(device)
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True

        elif flag == 'ViT-Large':   #BROKEN FIX !
            TempTransforms = transforms.Compose([transforms.Resize((224,224)),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(degrees=15),
                                                 #transforms.RandomResizedCrop()
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                                      std=[0.26862954, 0.26130258, 0.27577711])

                                                 ])
            self.transforms = TempTransforms
            model_name = 'vit_large_patch14_clip_224.openai_ft_in12k_in1k'
            #FeatureExtractor = ViTFeatureExtractor(model_name)
            self.model = timm.create_model(model_name,pretrained=True,num_classes=self.NumClasses)
            self.model.to(device)
            self.model.train()
            #model.head = nn.Linear(model.head.in_features,self.NumClasses)
            for param in self.model.parameters():
                param.requires_grad = True

        # elif flag == 'SwinV2-s':
        #     self.model = Swinv2ForImageClassification.from_pretrained('microsoft/swinv2-small-patch4-window8-256',num_labels=15)
        #     pass
        else:
            self.model = model
            self.device = device
            model.eval()
            model.to(self.device)
        #trainData = TrainSplit(self.transforms)
        #testData = TestSplit(self.transforms)
        ##Get weights for imbalance
        Weights = []
        TempDict = ClassImbalanceHelper()
        for key in classEncoding:
            Weights.append(TempDict[key])
        self.ImbalanceWeights = torch.tensor(Weights)
        self.ImbalanceWeights = self.ImbalanceWeights.to(device)

        #self.trainLoader = DataLoader(trainData, batch_size=256, shuffle=True, pin_memory=True)
        #self.valLoader = DataLoader(testData, batch_size=256, shuffle=True, pin_memory=True)  #consider not shuffling

    def LoadModelFromCheckpoint(self,Name):

        CheckPoint = torch.load(f'Models/{Name}')

        if self.flag == 'MbNv2':
            self.model = models.mobilenet_v2(pretrained=True).to(device)
            self.model.classifier[1] = nn.Linear(self.model.last_channel, self.NumClasses).to(device)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'ViT-Base':

            self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=self.NumClasses)

            self.model.load_state_dict(CheckPoint['state_dict'])


        elif self.flag == 'ViT-Large':
            self.model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224-in21k', num_labels=self.NumClasses)

            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'Swin-V2-Base':
            model_name = 'hf_hub:timm/swinv2_base_window12to16_192to256.ms_in22k_ft_in1k'
            # model_name = 'hf_hub:timm/swinv2_base_window12to24_192to256.ms_in22k_ft_in1k'
            self.model = timm.create_model(model_name, pretrained=False, num_classes=self.NumClasses).to(device)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'ENet-b5':
            model_name = 'efficientnet_b5'
            self.model = timm.create_model(model_name, pretrained=True).to(device)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, self.NumClasses).to(device)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'DenseNet-121':
            model_name = 'densenet121.tv_in1k'
            self.model = timm.create_model(model_name, pretrained=False, num_classes=self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])

        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        optimizer.load_state_dict(CheckPoint['optimizer'])
        epoch = CheckPoint['epoch']

        return (self.model,optimizer,epoch)



    def SaveCheckpoint(self,epoch,optimizer,model,Name):


        Location = f'TrainingRuns/{Name}.pt'
        state = {'epoch' : epoch,
                 'state_dict' : model.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                 }
        torch.save(state,Location)


    def TrainCustomSplit(self,percentage,batch_size,DebugVal = 1):
        #NewData = DataSetHelper(self.transforms)

        NewData = DataSetHelperMini(self.transforms,100)           #Mini dataset of 1000 (for my bad laptop)
        labels = np.array(NewData.labels)

        #Split for (train,val) and test (ensures test is seperate)
        train_val_idx,test_idx = train_test_split(range(len(NewData)), test_size=0.1, stratify=labels, random_state=42)

        train_idx,val_idx = train_test_split(train_val_idx,test_size=0.11,stratify=labels[train_val_idx],random_state=42)

        train_dataset = Subset(NewData,train_idx)

        ValidationTransforms = transforms.Compose([t for t in self.transforms.transforms if ((not isinstance(t,transforms.RandomHorizontalFlip)) or (not isinstance(t,transforms.RandomRotation)))])


        testData = DataSetHelper(ValidationTransforms)
        valData = DataSetHelper(ValidationTransforms)
        val_dataset = Subset(valData,val_idx)    #Remove random Flip !

        test_dataset = Subset(testData,test_idx)  #Remove Random Flip !



        # train_size = int(percentage * len(NewData))
        # test_size = len(NewData) - train_size
        # ##Stratified split
        #
        # train_data,test_data = torch.utils.data.random_split(NewData,[train_size,test_size])
        # self.trainLoader = DataLoader(train_data,batch_size=batch_size,shuffle=True,pin_memory=True)
        self.trainLoader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=2)       #worker is a core
        self.valLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,num_workers=2)
        self.testLoader = DataLoader(test_dataset,batch_size,shuffle=False)

    def TrainOfficialSplit(self,percentage,batch_size):
        if self.multiLabel == False:

            TrainValData = TrainSplit(self.transforms)
        else:
            TrainValData = TrainSplitMultiLabel(self.transforms)

        labels = np.array(TrainValData.trainingLabels)

        ValidationTransforms = transforms.Compose(
            [t for t in self.transforms.transforms if ((not isinstance(t, transforms.RandomHorizontalFlip)) or (not isinstance(t,transforms.RandomRotation)))])
        if self.multiLabel == False:

            ValData = TrainSplit(self.transforms)
        else:
            ValData = TrainSplitMultiLabel(self.transforms)
        #Stratify is removed for now
        train_idx, val_idx = train_test_split(range(len(TrainValData)), test_size=0.11,
                                              random_state=42)
        TrainData = Subset(TrainValData,train_idx)
        ValData = Subset(ValData,val_idx)
        if self.multiLabel == False:

            testData = TestSplit(ValidationTransforms)
        else:
            testData = TestSplitMultiLabel(ValidationTransforms)
        self.trainLoader = DataLoader(TrainData, batch_size=batch_size, shuffle=True, pin_memory=True,
                                      num_workers=9)
        self.valLoader = DataLoader(ValData, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3)
        self.testLoader = DataLoader(testData, batch_size=batch_size, shuffle=False)

    def TrainOfficial(self, epochs,resume=False):           #Let resume be False or a directory of a model to resume training from e.g. 'Models/MobileNetV24.pt

        if resume != False:

            Data = self.LoadModelFromCheckpoint(resume)
            start = Data[2]
            optimizer = Data[1]
        else:
            if not(os.path.exists('TrainingRuns')):
                os.mkdir('TrainingRuns')
            self.RunDirectory = str(datetime.now().strftime("%Y%m%d-%H%M%S") + self.flag)
            os.mkdir(f'TrainingRuns/{self.RunDirectory}')
            os.mkdir(f'TestRuns/{self.RunDirectory}')
            start = 0

            if self.endToEnd == True:
                optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
            else:

                optimizer = optim.Adam(self.model.classifier[1].parameters(), lr=0.0005)

        print(len(self.trainLoader))
        if self.multiLabel == False:

            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        else:
            #self.criterion = nn.BCEWithLogitsLoss()
            self.criterion = ClassAveragedBCELoss()
        self.BestValidationLoss = 99999
        self.NoImproveNumber = 0
        for epoch in range(start,epochs):
            i = 0
            running_loss=0.0
            self.model.train()
            for inputs, labels in self.trainLoader:
                #print(f'batch:{i}')
                # print('loop start')

                inputs, labels = inputs.to(device), labels.to(device)
                # print('loaded to GPU')
                optimizer.zero_grad()
                outputs = self.model(inputs)

                if self.flag == 'ViT-Base' or self.flag == 'ViT-Large' and 1==1:
                    outputs = outputs.logits


                loss = self.criterion(outputs,labels)
                loss.backward()
                optimizer.step()

                # print('batch complete')
                running_loss += loss.item()
                #print(f'Running Loss: {running_loss}')
                i += 1
            epoch_loss = running_loss / len(self.trainLoader)
            NewFile = open(f'TrainingRuns/{self.RunDirectory}/Epoch{epoch}.txt', 'a')
            NewFile.writelines(f'TrainingLoss: {epoch_loss} \n')

            NewFile.close()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')
            if self.multiLabel:
                temp = self.ValidateMultiLabel(epoch,optimizer,self.RunDirectory)
            else:

                temp = self.Validate(epoch,optimizer,self.RunDirectory)
            if temp == False:
                print('Early Stop triggered !')
                if self.multiLabel:


                    self.TestReusltsMultiLabel()
                else:
                    self.TestReusltsMultiLabel()
                return
            elif temp == 'LR_update':
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / 2
        if self.multiLabel:

            self.TestReusltsMultiLabel()
        else:
            self.TestResults()

    def TestResults(self):

        #Load Best model (assume best model is in current directory titled (something-BEST)
        #Load Best Model here
        CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/{self.flag}-BEST.pt')
        if self.flag == 'MbNv2':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/MobileNetV2-BEST.pt')
            self.model = models.mobilenet_v2(pretrained=True).to(device)
            self.model.classifier[1] = nn.Linear(self.model.last_channel, self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])

        elif self.flag == 'ViT-Small':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/ViT-Small-BEST.pt')
            model_name = 'vit_small_patch16_224.augreg_in21k'

            self.model = timm.create_model(model_name, pretrained=True, num_classes=self.NumClasses)
            self.model.to(device)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'ViT-Base':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/ViT-Base-BEST.pt')
            model_name = 'vit_base_patch16_224.augreg_in21k'

            self.model = timm.create_model(model_name, pretrained=True, num_classes=self.NumClasses)
            self.model.to(device)
            self.model.load_state_dict(CheckPoint['state_dict'])

        elif self.flag == 'ViT-Large':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/ViT-Large-BEST.pt')
            New = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224-in21k', num_labels=self.NumClasses)

            self.model = New.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'ENet-b1':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/ENet-b1-BEST.pt')
            model_name = 'efficientnet_b1'
            self.model = timm.create_model(model_name, pretrained=True).to(device)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])

        elif self.flag == 'ENet-b4':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/ENet-b4-BEST.pt')
            model_name = 'efficientnet_b4'
            self.model = timm.create_model(model_name, pretrained=True)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'ENet-b5':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/ENet-b5-BEST.pt')
            model_name = 'efficientnet_b5'
            self.model = timm.create_model(model_name, pretrained=True).to(device)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'Swin-Base':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/Swin-Base-BEST.pt')
            model_name = 'swin_base_patch4_window7_224.ms_in22k_ft_in1k'

            self.model = timm.create_model(model_name, pretrained=False, num_classes=self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'Swin-Large':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/Swin-Large-BEST.pt')
            model_name ='swin_large_patch4_window7_224.ms_in22k_ft_in1k'

            self.model = timm.create_model(model_name,pretrained=False,num_classes=self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'Swin-V2-Base':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/Swin-V2-Base-BEST.pt')
            model_name = 'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k'
            self.model = timm.create_model(model_name, pretrained=False, num_classes=self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'Swin-V2-Large':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/Swin-V2-Large-BEST.pt')
            model_name = 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k'
            self.model = timm.create_model(model_name, pretrained=False, num_classes=self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])

        elif self.flag == 'DenseNet-121':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/DenseNet-121-BEST.pt')
            model_name = 'densenet121.tv_in1k'
            self.model = timm.create_model(model_name,pretrained=False,num_classes=self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'DenseNet-169':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/DenseNet-169-BEST.pt')
            model_name = 'densenet169.tv_in1k'
            self.model = timm.create_model(model_name,pretrained=False,num_classes=self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'DenseNet-201':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/Densenet-201-BEST-pt')
            model_name = 'densenet201.tv_in1k'
            self.model = timm.create_model(model_name,pretrained=False,num_classes=self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])

        self.model.eval()
        self.model.to(device)
        true_labels = []
        predictions = []
        outputList = []
        inputList = []

        with torch.no_grad():
            for inputs,labels in self.testLoader:
                inputs = inputs.to(device)
                labels = labels.to(device)


                #outputs = self.model(inputs)
                if not(self.multiLabel):

                    if self.flag == 'ViT-Base' and 1==2:

                        outputs = SoftMaxPrediction(self.model,inputs,vit=False)
                    else:
                        outputs = SoftMaxPrediction(self.model, inputs, vit=False)

                else:
                    outputs = SigmoidPrediction(self.model,inputs)
                predicted = outputs.argmax(dim=1)
                inputList.extend(labels.cpu().numpy())
                labels = labels.argmax(dim=1)

                outputList.extend(outputs.cpu().numpy())
                predictions.extend(predicted.view(-1).cpu().numpy())
                true_labels.extend(labels.view(-1).cpu().numpy())

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        accuracy = sklearn.metrics.accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels,predictions,average='macro')
        recall = recall_score(true_labels,predictions,average='macro')
        f1 = f1_score(true_labels,predictions,average='macro')

        #AUC = roc_auc_score(true_labels,outputList,multi_class='ovr',average='macro')
        AUC = roc_auc_score(inputList, outputList,average='weighted', multi_class='ovo')


        #fpr,tpr,_ = roc_curve(true_labels,predictions)



        conf_matrix = confusion_matrix(true_labels,predictions)
        AUCDict = {}
        # for i in range(self.NumClasses):
        #     tempAUC = roc_auc_score(true_labels[:,i],predictions[:,i])
        #     AUCDict[list(classEncoding.keys())] = tempAUC
        #TEMPORARY !!!!!
        NewFile = open(f'TestRuns/DenseNETTEMP/FinalScore.txt','a')
        NewFile.writelines(f'Accuracy: {accuracy}\n')
        NewFile.writelines(f'Precision: {precision}\n')
        NewFile.writelines(f'Recall: {recall}\n')
        NewFile.writelines(f'F1 Score: {f1}\n')
        NewFile.writelines(f'AUC Score: {AUC}\n')
        #NewFile.writelines(f'IndividualClassAUC: {str(AUCDict)}\n')
        # NewFile.writelines(conf_matrix)
        print(conf_matrix)
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print(f'AUC Score {AUC}')
        #print(AUCDict)

    def TestReusltsMultiLabel(self):
        def find_optimal_thresholds(y_true,y_scores):
            optimalThresholds = {}
            for i in range(y_true.shape[1]):
                #precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_scores[:, i])
                precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_scores[:, i])
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)  # Avoid division by zero
                max_index = np.argmax(f1_scores)  # Find the threshold that gives the highest F1 score
                optimalThresholds[i] = (thresholds[max_index], f1_scores[max_index])
            return optimalThresholds

        def apply_thresholds(probabilities,thresholds):
            predictions = np.zeros_like(probabilities)
            for i ,threshold in enumerate(thresholds.values()):
                predictions[:,i] = (probabilities[:,i] >= threshold[0]).astype(int)
            return predictions
        #CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/{self.flag}-BEST.pt')
        if self.flag == 'MbNv2':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/MobileNetV2-BEST.pt')
            self.model = models.mobilenet_v2(pretrained=True).to(device)
            self.model.classifier[1] = nn.Linear(self.model.last_channel, self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'ViT-Base':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/ViT-Base-BEST.pt')
            model_name = 'vit_base_patch16_224.augreg_in21k'

            self.model = timm.create_model(model_name, pretrained=True, num_classes=self.NumClasses)
            self.model.to(device)

            self.model.load_state_dict(CheckPoint['state_dict'])

        elif self.flag == 'ViT-Small':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/ViT-Small-BEST.pt')
            model_name = 'vit_small_patch16_224.augreg_in21k'

            self.model = timm.create_model(model_name, pretrained=True, num_classes=self.NumClasses)
            self.model.to(device)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'ViT-Large':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/ViT-Large-BEST.pt')
            New = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224-in21k',
                                                            num_labels=self.NumClasses)

            self.model = New.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'ENet-b0':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/ENet-b0-BEST.pt')
            model_name = 'efficientnet_b0'
            self.model = timm.create_model(model_name, pretrained=True).to(device)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, self.NumClasses).to(device)
            self.model.load_state_dict(CheckPoint['state_dict'])
            for param in self.model.parameters():
                param.requires_grad = True
        elif self.flag == 'ENet-b1':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/ENet-b1-BEST.pt')
            model_name = 'efficientnet_b1'
            self.model = timm.create_model(model_name, pretrained=True).to(device)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])

        elif self.flag == 'ENet-b4':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/ENet-b4-BEST.pt')
            model_name = 'efficientnet_b4'
            self.model = timm.create_model(model_name, pretrained=True)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'ENet-b5':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/ENet-b5-BEST.pt')
            model_name = 'efficientnet_b5'
            self.model = timm.create_model(model_name, pretrained=True).to(device)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'Swin-Small':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/Swin-Small-BEST.pt')
            model_name = 'swin_small_patch4_window7_224.ms_in22k'
            self.model = timm.create_model(model_name, pretrained=True, num_classes=self.NumClasses).to(device)
            self.model.load_state_dict(CheckPoint['state_dict'])

        elif self.flag == 'Swin-Base':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/Swin-Base-BEST.pt')
            model_name = 'swin_base_patch4_window7_224.ms_in22k_ft_in1k'

            self.model = timm.create_model(model_name, pretrained=False, num_classes=self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'Swin-Large':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/Swin-Large-BEST.pt')
            model_name = 'swin_large_patch4_window7_224.ms_in22k_ft_in1k'

            self.model = timm.create_model(model_name, pretrained=False, num_classes=self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'Swin-V2-Base':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/Swin-V2-Base-BEST.pt')
            model_name = 'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k'
            self.model = timm.create_model(model_name, pretrained=False, num_classes=self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'Swin-V2-Large':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/Swin-V2-Large-BEST.pt')
            model_name = 'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k'
            self.model = timm.create_model(model_name, pretrained=False, num_classes=self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])

        elif self.flag == 'DenseNet-121':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/DenseNet-121-BEST.pt')
            model_name = 'densenet121.tv_in1k'
            self.model = timm.create_model(model_name, pretrained=False, num_classes=self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'DenseNet-169':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/DenseNet-169-BEST.pt')
            model_name = 'densenet169.tv_in1k'
            self.model = timm.create_model(model_name, pretrained=False, num_classes=self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])
        elif self.flag == 'DenseNet-201':
            CheckPoint = torch.load(f'TrainingRuns/{self.RunDirectory}/Densenet-201-BEST-pt')
            model_name = 'densenet201.tv_in1k'
            self.model = timm.create_model(model_name, pretrained=False, num_classes=self.NumClasses)
            self.model.load_state_dict(CheckPoint['state_dict'])
        self.model.eval()
        self.model.to(device)
        all_predictions = []
        all_labels = []
        all_probabilities = []
        runningLoss = 0.0
        with torch.no_grad():
            for inputs, labels in self.testLoader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                runningLoss += loss.item()
                predictions = torch.sigmoid(outputs) > 0.5
                probabilities = torch.sigmoid(outputs)
                predictions = predictions.cpu().numpy()
                labels = labels.cpu().numpy()
                probabilities = probabilities.cpu().numpy()
                all_predictions.append(predictions)
                all_probabilities.append(probabilities)
                all_labels.append(labels)
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        all_probabilities = np.vstack(all_probabilities)
        optimalThresholds = find_optimal_thresholds(all_labels, all_probabilities)
        ThresholdedPrecitions = apply_thresholds(all_probabilities, optimalThresholds)
        all_predictions = ThresholdedPrecitions
        validationLoss = runningLoss / len(self.valLoader)
        accuracy = sklearn.metrics.accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='samples')
        recall = recall_score(all_labels, all_predictions, average='samples')
        f1 = f1_score(all_labels, all_predictions, average='samples')
        auc = roc_auc_score(all_labels, all_probabilities, average='weighted', multi_class='ovr')
        conf_matrix = sklearn.metrics.multilabel_confusion_matrix(all_labels, all_predictions)
        report = classification_report(all_labels,all_predictions,output_dict=True)
        Reportdf = pd.DataFrame(report).transpose()
        Reportdf.to_csv(f'TestRuns/{self.RunDirectory}/classification_report.csv')
        PicklePath = f'TestRuns/{self.RunDirectory}/AUCs.pickle'
        aucScores = roc_auc_score(all_labels,all_probabilities,average=None)
        with open(PicklePath,'wb') as file:
            pickle.dump(aucScores,file)
        NewFile = open(f'TestRuns/{self.RunDirectory}/FinalScore.txt', 'a')
        #NewFile = open(f'TestRuns/DenseNETTEMP/FinalScore.txt', 'a')
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print("Multilabel Confusion Matrix:")
        print(conf_matrix)
        NewFile.writelines(f'ValidationLoss: {validationLoss}\n')
        NewFile.writelines(f'Accuracy: {accuracy}\n')
        NewFile.writelines(f'Precision: {precision}\n')
        NewFile.writelines(f'Recall: {recall}\n')
        NewFile.writelines(f'F1 Score: {f1}\n')
        NewFile.writelines(f'AUC Score: {auc}\n')
    def ValidateMultiLabel(self,epoch,optimizer,Location):
        def find_optimal_thresholds(y_true,y_scores):
            optimalThresholds = {}
            for i in range(y_true.shape[1]):
                #precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_scores[:, i])
                precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_scores[:, i])
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)  # Avoid division by zero
                max_index = np.argmax(f1_scores)  # Find the threshold that gives the highest F1 score
                optimalThresholds[i] = (thresholds[max_index], f1_scores[max_index])
            return optimalThresholds

        def apply_thresholds(probabilities,thresholds):
            predictions = np.zeros_like(probabilities)
            for i ,threshold in enumerate(thresholds.values()):
                predictions[:,i] = (probabilities[:,i] >= threshold[0]).astype(int)
            return predictions
        MLclassEncoding = {
            'Atelectasis': torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'Consolidation': torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'Infiltration': torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'Pneumothorax': torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'Edema': torch.FloatTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'Emphysema': torch.FloatTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            'Fibrosis': torch.FloatTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
            'Effusion': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
            'Pneumonia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
            'Pleural_Thickening': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
            'Cardiomegaly': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
            'Nodule': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
            'Hernia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
            'Mass': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
            # 'No Finding': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        }
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        runningLoss = 0.0
        with torch.no_grad():
            for inputs,labels in self.valLoader:

                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                runningLoss += loss.item()
                predictions = torch.sigmoid(outputs) > 0.5
                probabilities = torch.sigmoid(outputs)
                predictions = predictions.cpu().numpy()
                labels = labels.cpu().numpy()
                probabilities = probabilities.cpu().numpy()
                all_predictions.append(predictions)
                all_probabilities.append(probabilities)
                all_labels.append(labels)
        print(probabilities)
        all_predictions = np.vstack(all_predictions)
        all_labels = np.vstack(all_labels)
        all_probabilities = np.vstack(all_probabilities)

        validationLoss = runningLoss / len(self.valLoader)
        optimalThresholds = find_optimal_thresholds(all_labels,all_probabilities)
        ThresholdedPrecitions = apply_thresholds(all_probabilities,optimalThresholds)
        all_predictions = ThresholdedPrecitions
        accuracy = sklearn.metrics.accuracy_score(all_labels,all_predictions)
        precision = precision_score(all_labels,all_predictions,average ='samples')
        recall = recall_score(all_labels,all_predictions,average='samples')
        f1 = f1_score(all_labels,all_predictions,average='samples')
        auc = roc_auc_score(all_labels,all_probabilities,average='weighted',multi_class='ovr')
        conf_matrix = sklearn.metrics.multilabel_confusion_matrix(all_labels,all_predictions)
        Labels = list(MLclassEncoding.keys())
        print(classification_report(all_labels,all_predictions,target_names=Labels))
        NewFile = open(f'TrainingRuns/{Location}/Epoch{epoch}.txt', 'a')

        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print("Multilabel Confusion Matrix:")
        print(conf_matrix)
        NewFile.writelines(f'ValidationLoss: {validationLoss}\n')
        NewFile.writelines(f'Accuracy: {accuracy}\n')
        NewFile.writelines(f'Precision: {precision}\n')
        NewFile.writelines(f'Recall: {recall}\n')
        NewFile.writelines(f'F1 Score: {f1}\n')
        NewFile.writelines(f'AUC Score: {auc}\n')

        if validationLoss < self.BestValidationLoss:
            self.NoImproveNumber = 0
            self.BestValidationLoss = validationLoss
            #JOE FIX THIS !!! ADD SAVING MODEL FOR EFFICIENTNET !
            if self.flag == 'MbNv2':

                self.SaveCheckpoint(epoch, optimizer, self.model, f'{Location}/MobileNetV2-BEST')

            elif self.flag == 'ViT-Small':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/ViT-Small-BEST')
            elif self.flag == 'ViT-Base':
                self.SaveCheckpoint(epoch, optimizer, self.model, f'{Location}/ViT-Base-BEST')

            elif self.flag == 'ViT-Large':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/ViT-Large-BEST')
            elif self.flag == 'DenseNet-121':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/DenseNet-121-BEST')
            elif self.flag == 'DenseNet-169':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/DenseNet-169-BEST')
            elif self.flag == 'DenseNet-201':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/DenseNet-201-BEST')
            elif self.flag == 'ENet-b0':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/ENet-b0-BEST')
            elif self.flag == 'ENet-b1':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/ENet-b1-BEST')
            elif self.flag == 'ENet-b4':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/ENet-b4-BEST')
            elif self.flag == 'ENet-b5':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/ENet-b5-BEST')
            elif self.flag == 'Swin-Small':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/Swin-Small-BEST')
            elif self.flag == 'Swin-Base':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/Swin-Base-BEST')
            elif self.flag == 'Swin-Large':
                self.SaveCheckpoint(epoch, optimizer, self.model, f'{Location}/Swin-Large-BEST')
            elif self.flag == 'Swin-V2-Base':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/Swin-V2-Base-BEST')
            elif self.flag == 'Swin-V2-Large':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/Swin-V2-Large-BEST')
        else:
            self.NoImproveNumber += 1

        NewFile.close()
        if self.NoImproveNumber >= 3:
            if self.NoImproveNumber >= 5:

                return False
            else:
                return 'LR_update'
    def Validate(self,epoch,optimizer,Location):

        self.model.eval()
        true_labels = []
        predictions = []
        outputList = []
        inputList = []
        runningLoss = 0.0
        with torch.no_grad():

            for inputs,labels in self.valLoader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputsForLoss = self.model(inputs)
                if self.flag == 'ViT-Base' and 1==2:
                    outputsForLoss = outputsForLoss.logits

                loss = self.criterion(outputsForLoss,labels)
                runningLoss += loss.item()

                if not(self.multiLabel):
                    if self.flag == 'ViT-Base' and 1 == 2:

                        outputs = SoftMaxPrediction(self.model, inputs, vit=False)
                    else:
                        outputs = SoftMaxPrediction(self.model, inputs, vit=False)
                else:
                    outputs = SigmoidPrediction(self.model,inputs)

                predicted = outputs.argmax(dim=1)
                inputList.extend(labels.cpu().numpy())
                labels = labels.argmax(dim=1)

                outputList.extend(outputs.cpu().numpy())
                predictions.extend(predicted.view(-1).cpu().numpy())
                true_labels.extend(labels.view(-1).cpu().numpy())
        validationLoss = runningLoss / len(self.valLoader)
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        accuracy = sklearn.metrics.accuracy_score(true_labels,predictions)
        
        precision = precision_score(true_labels,predictions,average='macro')
        recall = recall_score(true_labels,predictions,average='macro')
        f1 = f1_score(true_labels,predictions,average='macro')

        #AUC = roc_auc_score(true_labels,outputList,multi_class='ovr',average='macro')
        AUC = roc_auc_score(inputList, outputList,average='weighted', multi_class='ovo')


        #fpr,tpr,_ = roc_curve(true_labels,predictions)

        if validationLoss < self.BestValidationLoss:
            self.NoImproveNumber = 0
            self.BestValidationLoss = validationLoss
            #JOE FIX THIS !!! ADD SAVING MODEL FOR EFFICIENTNET !
            if self.flag == 'MbNv2':

                self.SaveCheckpoint(epoch, optimizer, self.model, f'{Location}/MobileNetV2-BEST')
            elif self.flag == 'ViT-Base':
                self.SaveCheckpoint(epoch, optimizer, self.model, f'{Location}/ViT-Base-BEST')

            elif self.flag == 'ViT-Large':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/ViT-Large-BEST')
            elif self.flag == 'DenseNet-121':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/DenseNet-121-BEST')
            elif self.flag == 'DenseNet-169':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/DenseNet-169-BEST')
            elif self.flag == 'DenseNet-201':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/DenseNet-201-BEST')
            elif self.flag == 'ENet-b1':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/ENet-b1-BEST')
            elif self.flag == 'ENet-b4':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/ENet-b4-BEST')
            elif self.flag == 'ENet-b5':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/ENet-b5-BEST')
            elif self.flag == 'Swin-Base':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/Swin-Base-BEST')
            elif self.flag == 'Swin-Large':
                self.SaveCheckpoint(epoch, optimizer, self.model, f'{Location}/Swin-Large-BEST')
            elif self.flag == 'Swin-V2-Base':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/Swin-V2-Base-BEST')
            elif self.flag == 'Swin-V2-Large':
                self.SaveCheckpoint(epoch,optimizer,self.model,f'{Location}/Swin-V2-Large-BEST')
        else:
            self.NoImproveNumber += 1

        AUCDict = {}

        conf_matrix = confusion_matrix(true_labels,predictions)
        print(conf_matrix)
        NewFile = open(f'TrainingRuns/{Location}/Epoch{epoch}.txt','a')

        NewFile.writelines(f'ValidationLoss: {validationLoss}\n')
        NewFile.writelines(f'Accuracy: {accuracy}\n')
        NewFile.writelines(f'Precision: {precision}\n')
        NewFile.writelines(f'Recall: {recall}\n')
        NewFile.writelines(f'F1 Score: {f1}\n')
        NewFile.writelines(f'AUC Score: {AUC}\n')
        NewFile.writelines(f'IndividualClassAUC: {str(AUCDict)}\n')
        # NewFile.writelines(conf_matrix)
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print(f'AUC Score {AUC}')

        NewFile.close()
        if self.NoImproveNumber >= 3:
            if self.NoImproveNumber >= 5:


                return False
            else:
                return 'LR_update'
        #print(f'Confusion Matrix: \n {conf_matrix}')
    def LocalisationStudy(self):
        IoUs = {}
        for i in range(0,984):
            print(i)
            CurrentData = LocalisationFromIndex(i)
            finding = CurrentData[1]
            if finding == 'Infiltrate':
                finding = 'Infiltration'
            if finding not in list(IoUs.keys()):
                IoUs[finding] = []
            labelEncoding = classEncoding[finding]
            LabelNumber = int(np.argmax(labelEncoding))
            x,y,w,h = CurrentData[2][0],CurrentData[2][1],CurrentData[2][2],CurrentData[2][3]
            CurrentFileName = CurrentData[0]
            CurrentImage = Image.open(CurrentFileName)
            CurrentImage = CurrentImage.convert('RGB')

            ImageTensor = self.transforms(CurrentImage)
            ImageTensor.to(device)
            ImageTensor = ImageTensor.unsqueeze(0)
            Map = CAMMethods(self.model,ImageTensor,0.1,[ClassifierOutputTarget(LabelNumber)])
            ImageDimension = len(Map)
            ScalingFactor = ImageDimension / 1024
            x,y,w,h = int(x * ScalingFactor),int(y*ScalingFactor),int(w*ScalingFactor),int(h*ScalingFactor)
            MapMask = Map[:]

            for i in range(ImageDimension):
                for k in range(ImageDimension):
                    if MapMask[i][k] != 0:
                        MapMask[i][k] = 1


            #Intersection over union
            rectangleMask = np.zeros((len(Map),len(Map)))
            rectangleMask[y:y+h,x:x+w] = 1
            intersection = np.sum(rectangleMask * MapMask)

            unionMask = rectangleMask + MapMask
            unionMask[unionMask>1] = 1
            union = np.sum(unionMask)
            IoU = intersection/union
            IoUs[finding].append(IoU)
            #print(IoU)
            #print(IoUs)
            #ShowGradCam(CurrentImage,Map)
            #print('test')
            #time.sleep(5.0)
        for key in list(IoUs.keys()):
            print(f'Average {key}: {sum(IoUs[key]) / len(IoUs[key])}')
        # TestImagePath = Test
        # TestImage = Image.open(TestImagePath)
        # TestImage = TestImage.convert('RGB')
        #
        # inputImage = testTransforms(TestImage)
        # inputImage.to(device)
        # inputImage = inputImage.unsqueeze(0)
        # Map = CAMMethods(test1.model, inputImage, 0.5)
        # ShowGradCam(TestImage, Map)

class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self,model):
        super(HuggingfaceToTensorModelWrapper,self).__init__()
        self.model = model
    def forward(self,x):
        return self.model(x).logits
class GradCamHelper:

    def __init__(self,model,CamLayer='features'):
        self.model = model
        self.model = self.model.to('cpu')
        ##IF VIT !!!
        self.model = HuggingfaceToTensorModelWrapper(self.model)

        self.GradCAM = GradCAM(model,CamLayer)

    def Analyze(self,input_tensor,input_image):

        #input_tensor = input_tensor.unsqueeze(0)
        out = self.model(input_tensor)
        #out = out.logits
        cam = self.GradCAM(out.squeeze(0).argmax().item(),out)




        HeatMap = cam[0]
        fig, ax = plt.subplots()
        ax.axis('off')
        input_image = input_image.resize((224,224))
        ax.imshow(input_image)

        overlay = torchvision.transforms.functional.to_pil_image(HeatMap.detach(),mode='F').resize((224,224),resample=PIL.Image.BICUBIC)

        cmap = matplotlib.colormaps['jet']
        overlay = (255 * cmap(np.asarray(overlay) ** 2)[:,:,:3]).astype(np.uint8)
        overlay = torchvision.transforms.functional.to_pil_image(overlay)
        newImage = Image.blend(input_image,overlay,0.3)
        #ax.imshow(overlay,alpha=0.5,interpolation='nearest',extent='extent')
        ax.imshow(newImage)
        plt.show()
        #result = overlay_mask(Image.fromarray(input_tensor.squeeze(0).permute(1, 2, 0).byte().numpy()), Image.fromarray(heatmap))
        #plt.imshow(input_image)
        #plt.imshow(heatmap,alpha=0.5,cmap='jet')

def display_tensor_as_image(tensor):

    if tensor.ndim == 4 and tensor.shape[0] == 1:  # Handle batch dimension if present
        tensor = tensor.squeeze(0)

    # Convert tensor to numpy array
    try:

        numpy_img = tensor.cpu().detach().numpy()
    except:
        numpy_img = tensor
        pass

    if numpy_img.ndim == 3:
        numpy_img = np.transpose(numpy_img, (1, 2, 0))

    # Normalize if necessary
    if numpy_img.max() > 1.0:
        numpy_img = numpy_img / 255.0

    plt.imshow(numpy_img)
    plt.axis('off')  # Turn off axis numbers
    plt.show()
    return numpy_img


def PlotBoundingBox(ImageFile,x,y,w,h,Finding):
    #image = Image.open(fileName, 'r').convert('RGB')
    ImageData = Image.open(ImageFile,'r').convert('RGB')


    fig,ax = plt.subplots(1)
    plt.title(Finding)
    ax.imshow(ImageData)
    rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.show()

def LocalisationFromIndex(idx):
    CSV = pd.read_csv('Data/BBox_List_2017.csv')
    row = CSV.loc[idx]
    ImageName = f'Data/images_001/images/{row["Image Index"]}'
    print(ImageName)
    finding = row['Finding Label']
    x,y,w,h = row['Bbox [x'],row['y'],row['w'],row['h]']
    PlotBoundingBox(ImageName,x,y,w,h,finding)
    return (ImageName,finding,(x,y,w,h))
#PlotBoundingBox('Data/images_001/images/00013118_008.png',225.084745762712,547.019216763771,86.7796610169491,79.1864406779661)
def SigmoidPrediction(model,Input):
    model.eval()
    with torch.no_grad():
        logits = model(Input)
        probabilities = nn.functional.sigmoid(logits)
    return probabilities
def SoftMaxPrediction(model,Input,vit):
    model.eval()
    if vit == False:

        with torch.no_grad():
            logits = model(Input)
            probabilities = nn.functional.softmax(logits,dim=1)
    else:
        with torch.no_grad():
            logits = model(Input)
            logits = logits.logits
            probabilities = nn.functional.softmax(logits,dim=1)
    return probabilities

def LocalisationEvaluation():
    #Loop Through the localisation data
    CSV = pd.read_csv('Data/BBox_List_2017.csv')
    for i in range(len(CSV)):
        LocalisationFromIndex(i)

    pass
    #Function to iterate through directory of textFiles to plot training run results.
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
def ProbabiltiyToLabel(output):
    Labels = list(classEncoding)
    DictionaryOutput = {}
    for item in Labels:
        DictionaryOutput[item] = 0

    output = output.cpu().numpy()
    for i in range(15):
        DictionaryOutput[Labels[i]] = output[0][i]
    return DictionaryOutput

def reshape_transform(tensor, height=8, width=8):

    # result = tensor[:, 1:, :].reshape(tensor.size(0),
    #                                        height, width, tensor.size(2))
    # result = result.transpose(2, 3).transpose(1, 2)
    result = tensor.permute(0,3,1,2)
    return result
def CAMMethods(model,image,Threshold=0.2,label=False):
    ##A function to take any model and generate class activation maps with parameters
    #Extend for eigen ablation and other versions of Grad-CAM
    classEncoding = {
        'Atelectasis': torch.FloatTensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Consolidation': torch.FloatTensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Infiltration': torch.FloatTensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Pneumothorax': torch.FloatTensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Edema': torch.FloatTensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Emphysema': torch.FloatTensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Fibrosis': torch.FloatTensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        'Effusion': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        'Pneumonia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        'Pleural_Thickening': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        'Cardiomegaly': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        'Nodule': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        'Hernia': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
        'Mass': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
        'No Finding': torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    }
    #SWIN Reshape !

    ##For VIT
    model.eval()
    model.to(device)
    #target_layers = [model.layers[-1].blocks[-1].norm1,model.layers[-2].blocks[-1].norm1]
    #target_layers = [model.conv_head]
    target_layers = [model.features[-1]]
    #cam = GradCAM(model=model,target_layers=target_layers,reshape_transform=reshape_transform)
    cam = GradCAM(model=model, target_layers=target_layers)
    cam.batch_size = 1

    targets = label
    #targets = [ClassifierOutputTarget(0)]
    #Change for different pathologies
    #targets = [classEncoding['Atelectasis']]
    #targets = [ClassifierOutputTarget(13)]
    targets = None
    #greyscaleCam.to(device)
    #targets = 10
    greyscaleCam = cam(input_tensor=image,targets=targets,aug_smooth=True,eigen_smooth=True)
    #display_tensor_as_image(greyscaleCam[0,:])
    greyscaleCam = greyscaleCam[0,:]
    for x in range(len(greyscaleCam)):
        for y in range(len(greyscaleCam[0])):
            item = greyscaleCam[x][y]
            if item < Threshold:
                greyscaleCam[x][y] = 0
    RGBImage = torchvision.transforms.functional.to_pil_image(image.squeeze(0))
    #visualisation = show_cam_on_image(RGBImage,greyscaleCam)
    return greyscaleCam


def ApplyColourMap(GradCam):
    GradCam = (GradCam - np.min(GradCam) / (np.max(GradCam) - np.min(GradCam)))

    colourMap = plt.get_cmap('jet')
    heatMap = colourMap(GradCam)[:,:,:3]
    heatMap = (heatMap * 255).astype(np.uint8)
    heatMap = Image.fromarray(heatMap)
    return heatMap

def ShowGradCam(image,GradCam):

    Image.blend(image.resize([224, 224]), ApplyColourMap(GradCam), alpha=0.3).show()



if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test1 = ModelProcess('Mob','Trans',device,'Swin-Base',multiLabel=True)
    test1.TrainOfficialSplit(0.5,64)
    test1.TrainOfficial(100)


    #CheckPoint = torch.load(f'Models/DenseNet-121-BEST.pt')
    #model_name = 'densenet121.tv_in1k'
    #test1.model = timm.create_model(model_name, pretrained=True, num_classes=14)
    #test1.model.load_state_dict(CheckPoint['state_dict'])
    #test1.model =test1.model.to(device)
    #test1.criterion = ClassAveragedBCELoss()
    #test1.ValidateMultiLabel(1,2,3)
    #test1.LocalisationStudy()
    #LocalisationEvaluation()
    #test1.LoadModelFromCheckpoint('DenseNet-121-BEST.pt')
    #test1.TestResults()
    #test1.LocalisationStudy()

    #test1.TestResults()
    #
    # test1.model.eval()
    # # testTransforms = transforms.Compose([transforms.Resize((256, 256)),
    # #                                     # transforms.RandomHorizontalFlip(),
    # #                                     transforms.ToTensor(),
    # #                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    # #                                                          std=[0.229, 0.224, 0.225])
    # #
    # #                                     ])
    testTransforms = transforms.Compose([transforms.Resize((224, 224)),
                                                         #transforms.RandomHorizontalFlip(),
                                                         #transforms.RandomRotation(degrees=15),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225])

                                                         ]
                                                        )
    #
    Test = LocalisationFromIndex(300)
    # TestImagePath = Test
    # TestImage = Image.open(TestImagePath[0])
    # TestImage = TestImage.convert('RGB')
    # #
    # inputImage = testTransforms(TestImage)
    # inputImage.to(device)
    # inputImage = inputImage.unsqueeze(0)
    # Map = CAMMethods(test1.model,inputImage,0.5)
    # ShowGradCam(TestImage,Map)
    #ProbabiltiyToLabel(SoftMaxPrediction(test1.model,inputImage,True))
    #GradCamExpriement = GradCamHelper(test1.model,test1.model.vit.encoder.layer[-1].attention)

    #Cam = GradCamExpriement.Analyze(inputImage, TestImage)
#    pass


#
#
#
#
#
#
#
# #Grad-Cam Experiment
#
# #Grad-Cam map generated
#
# #fix resizing heatmap as overlay  DOne !
#
#
testTransforms = transforms.Compose([transforms.Resize((224, 224)),
                                                         # transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])

                                                       ])
# #
# #
# #
# #
# #
# LoadedModel = test1.LoadModelFromCheckpoint('MobileNetV210.pt')
# #
# # #test1.Validate(-1)
# #
#
# Test = LocalisationFromIndex(0)
# TestImagePath = Test
# TestImage = Image.open(TestImagePath)
# TestImage = TestImage.convert('RGB')
#
# inputImage = testTransforms(TestImage)
# inputImage.to(device)
# inputImage.unsqueeze(0)
# Tester = inputImage.to(device)
# Tester = Tester.unsqueeze(0)
# print(SoftMaxPrediction(LoadedModel[0],Tester))
#
# GradCamExpriement = GradCamHelper(LoadedModel[0])
#
# Cam = GradCamExpriement.Analyze(inputImage,TestImage)
#

