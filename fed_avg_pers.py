
"""
Created on Thu Aug 18 11:58:15 2022

@author: rahulbhope
"""

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import argparse
import os
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import torchvision.transforms as transforms
import torch.utils.data as data
from itertools import product
import copy
from sklearn.metrics import confusion_matrix
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision import models,transforms


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.utils.data as data
import pandas as pd
from PIL import Image

import logging
import os
import json
#import h5py
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
#import seaborn as sns
import matplotlib.pyplot as plt
import glob
import time
import re
from joblib import Parallel, delayed
import pickle
import warnings
from sklearn.cluster import DBSCAN
import itertools
from sklearn.cluster import OPTICS
from scipy.cluster.vq import vq, kmeans2, whiten
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.special import rel_entr
from sklearn.cluster import SpectralClustering
import csv
import sys
import random
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

how_many_parts = 0#int(sys.argv[1])
alpha = 0#float(sys.argv[2])
#(X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts) = partition_data('cifar10','/Users/rahulbhope/Documents/rahul_code/cifar','/Users/rahulbhope/Documents/rahul_code/cifar','hetero-dir',100,alpha=alpha)
#train_dl, test_dl = get_dataloader('cifar10', '/Users/rahulbhope/Documents/rahul_code/cifar', 8, 8, dataidxs=net_dataidx_map[1])
how_many_clients = 50
how_many_labels = 7
train_bs = 8
test_bs = 8
#alpha = 0
num_clients = 50

input_size = 224

class CustomDataset(data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)
        return X, y


def record_net_data_stats(y_train, net_dataidx_map, logdir):

	net_cls_counts = {}

	for net_i, dataidx in net_dataidx_map.items():
		unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
		tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
		net_cls_counts[net_i] = tmp

	logging.debug('Data statistics: %s' % str(net_cls_counts))

	return net_cls_counts



def partition_dataset(alpha, y_train, num_clients):
    #alpha = 1
    num_users = num_clients
    num_classes = len(np.unique(y_train))
    num_datapoints = len(y_train)

    active_classes = [ _ for _ in range(num_classes)]

    samples_per_user = int(y_train[:num_datapoints].shape[0]/num_users)
    samples_per_class = int(y_train[:num_datapoints].shape[0]/num_classes)
    user_dataidx_map = {}

    idxs_ascending_labels = np.argsort(y_train[:num_datapoints])
    labels_idx_map = np.zeros((num_classes, samples_per_class))

    labels_idx_maps1 = [[] for _ in range(10)]


    for i in active_classes:
        labels_idx_map[i] = idxs_ascending_labels[i*samples_per_class:(i+1)*samples_per_class]
        
        for idx_map in labels_idx_map[i]:
            labels_idx_maps1[i].append(int(idx_map))
        
        np.random.shuffle(labels_idx_map[i])
        
    for user_id in range(num_users):
        
        for k in active_classes:
            if len(labels_idx_maps1[k])<10:
                active_classes.remove(k)
                num_classes = num_classes - 1
        
        current_user_dataidx = []
        flag = True
        while flag:
            current_user_dataidx = []
            proportions = np.random.dirichlet(np.repeat(alpha, num_classes))
            histogram = samples_per_user*proportions
            histogram = histogram.astype(np.int)
            flag = False
            for i in active_classes:
                if histogram[active_classes.index(i)]>len(labels_idx_maps1[i]):
                    flag=True
        
        for i in active_classes:
            current_user_dataidx.append(labels_idx_maps1[i][:histogram[active_classes.index(i)]])
            for j in range(0,histogram[active_classes.index(i)]):
                labels_idx_maps1[i].pop(0)
            np.random.shuffle(labels_idx_map[i])
            
        user_dataidx_map[user_id] = np.hstack(current_user_dataidx).astype(np.int).flatten()
    '''    
    list_1 = []#[[0,0,0,0,0,0,0,0,0,0] for _ in range(num_users)]
    
    for _ in range(num_users):
        list_temp = []
        for j1 in range(num_classes):
            list_temp.append(0)
        list_1.append(list_temp)
    

    for key in user_dataidx_map:
        #print(key)
        for idx in user_dataidx_map[key]:
            labels = y_train[idx]
            list_1[key][labels] = list_1[key][labels] + 1

    for key in user_dataidx_map:
        for key1 in user_dataidx_map:
            if key!=key1:
                if len(np.intersect1d(user_dataidx_map[key], user_dataidx_map[key1])) > 0:
                    print(key,key1)
                    
    
    '''
    return user_dataidx_map


def partition_alpha_zero(trainset,testset,key_dir='cifar/key_alpha_0.csv'):
    list_img = []
    list_labels =[]
    list_idx = []

    for i in range(0,how_many_labels):
        list_idx.append([])


    for i in range(0, len(trainset)):
        list_img.append(trainset[i][0])
        list_labels.append(trainset[i][1])
        list_idx[trainset[i][1]].append(i)
        
        
    list_img_label = []

    for i in range(0,how_many_labels):
        list_img_label.append([])
        
    for i in range(0,len(list_labels)):
        list_img_label[list_labels[i]].append(list_img[i])
        

    dict_keys_order = pd.read_csv(key_dir)  
            
    list_new_keys = np.array(dict_keys_order['key'])


    start_with = 0
    for i in list_new_keys:
        
        if len(list_idx[start_with]) == 0:
            start_with = start_with + 1
        
        
        list_img_cli = []
        list_label_cli = []
                
        for k in range(0,500):
            list_img_cli.append(list_idx[start_with][0])
            list_label_cli.append(start_with)
            list_idx[start_with].pop(0)
        
        netdata_idx_map_train[int(i)] = list_img_cli
        
        
        
        
    list_img = []
    list_labels =[]
    list_idx = []

    for i in range(0,how_many_labels):
        list_idx.append([])


    for i in range(0, len(testset)):
        list_img.append(trainset[i][0])
        list_labels.append(trainset[i][1])
        list_idx[testset[i][1]].append(i)
        
        
    list_img_label = []

    for i in range(0,how_many_labels):
        list_img_label.append([])
        
    for i in range(0,len(list_labels)):
        list_img_label[list_labels[i]].append(list_img[i])
        



    dict_keys_order = pd.read_csv(key_dir)  
            
    list_new_keys = np.array(dict_keys_order['key'])


    start_with = 0
    for i in list_new_keys:
        
        if len(list_idx[start_with]) == 0:
            start_with = start_with + 1
        
        
        list_img_cli = []
        list_label_cli = []
                
        for k in range(0,100):
            list_img_cli.append(list_idx[start_with][0])
            list_label_cli.append(start_with)
            list_idx[start_with].pop(0)
        
        netdata_idx_map_test[int(i)] = list_img_cli
        
    return netdata_idx_map_train, netdata_idx_map_test


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(13456, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18, resnet34, resnet50, resnet101
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    elif model_name == "densenet":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size


#model_0 = Model().to(DEVICE)   

model_name = 'densenet'
num_classes = 7
feature_extract = False
input_size = 224
# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
# Define the device:
#device = torch.device('cuda:0')
# Put the model on the device:
model_0 = model_ft.to(DEVICE)


#model_0 = Net() 


def loss_classifier(predictions,labels):
    
    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss(reduction="mean")
    
    return loss(m(predictions) ,labels.view(-1))


def loss_dataset(model, dataset, loss_f):
    """Compute the loss of `model` on `dataset`"""
    loss=0
    
    with torch.no_grad():
        for idx,(features,labels) in enumerate(dataset):

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            
            predictions= model(features)
            loss+=loss_f(predictions,labels)
    
    loss/=idx+1
    return loss.cpu()


def accuracy_dataset(model, dataset):
    """Compute the accuracy of `model` on `dataset`"""
    
    correct=0

    with torch.no_grad():
    
        for features,labels in iter(dataset):

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            
            predictions= model(features)
            
            _,predicted=predictions.max(1,keepdim=True)
            
            correct+=torch.sum(predicted.view(-1,1)==labels.view(-1, 1)).item()
            
        accuracy = 100*correct/len(dataset.dataset)
        
    return accuracy


def accuracy_dataset_per_label(model, dataset,no_labels):
    """Compute the accuracy of `model` on `dataset`"""
    
    correct=0#[0] * no_labels

    label_list=[]
    predicted_list=[]

    with torch.no_grad():
    
        for features,labels in iter(dataset):

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            
            predictions= model(features)
            
            _,predicted=predictions.max(1,keepdim=True)
            correct+=torch.sum(predicted.view(-1,1)==labels.view(-1, 1)).item()
            labels = labels.to(DEVICE).cpu().detach().numpy()
            predicted = predicted.cpu().detach().numpy()
            label_list.append(labels)
            predicted_list.append(predicted)
            
    

    labels_arr = np.hstack(label_list)
    predicted_arr = np.hstack(np.vstack(predicted_list))



    #print(labels_arr, predicted_arr)

    
    cm_matrix = confusion_matrix(labels_arr,predicted_arr,labels=np.arange(0,no_labels,1))
    #print(cm_matrix)
    max_cm = np.amax(cm_matrix, axis=1).tolist()
    max_cm_2 = np.amax(cm_matrix, axis=0).tolist()
    cm_matrix_1 = cm_matrix.diagonal()/cm_matrix.sum(1)
        
    accuracy = 100*correct/len(dataset.dataset)
        
    return np.nan_to_num(cm_matrix_1), np.mean(np.nan_to_num(cm_matrix_1))

def train_step(model, model_0, mu:int, optimizer, train_data, loss_f):
    """Train `model` on one epoch of `train_data`"""
    
    total_loss=0
    
    for idx, (features,labels) in enumerate(train_data):

        features = features.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        predictions= model(features)
        
        loss=loss_f(predictions,labels)
        loss+=mu/2*difference_models_norm_2(model,model_0)
        total_loss+=loss
        
        loss.backward()
        optimizer.step()
        
    return total_loss/(idx+1)



def local_learning(model, mu:float, optimizer, train_data, epochs:int, loss_f):

    start = time.time()
    
    model_0=deepcopy(model)
    
    for e in range(epochs):
        local_loss=train_step(model,model_0,mu,optimizer,train_data,loss_f)

    print(time.time() - start)
        
    return float(local_loss.cpu().detach().numpy())


def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters
    """
    
    tensor_1=list(model_1.parameters())
    tensor_2=list(model_2.parameters())
    
    norm=sum([torch.sum((tensor_1[i]-tensor_2[i])**2) 
        for i in range(len(tensor_1))])
    
    return norm


def set_to_zero_model_weights(model):
    """Set all the parameters of a model to 0"""

    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data)




def average_models(model, clients_models_hist:list , weights:list):


    """Creates the new model of a given iteration with the models of the other
    clients"""
    
    new_model=deepcopy(model)
    set_to_zero_model_weights(new_model)

    for k,client_hist in enumerate(clients_models_hist):
        
        for idx, layer_weights in enumerate(new_model.parameters()):

            contribution=client_hist[idx].data*weights[k]
            layer_weights.data.add_(contribution)
            
    return new_model 





def FedProx(alpha,model, test_ld, training_sets:dict, n_iter:int, testing_sets:dict, mu=0, 
    file_name="test", epochs=5, lr=10**-2, decay=1,top=20):
    """ all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the 
            training set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularization term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration
    
    returns :
        - `model`: the final global model 
    """




        
    picked_model = []
    if alpha == 0.3:
        if how_many_parts == 15:
            picked_model = [[0, 11, 43, 14, 23, 9, 2, 31], [24, 45, 33, 14, 23, 15, 32, 35], [19, 47, 33, 14, 23, 15, 39, 35], [42, 10, 33, 14, 23, 15, 32, 37], [36, 41, 33, 14, 23, 9, 34, 31], [13, 40, 43, 14, 23, 15, 16, 35], [6, 10, 43, 14, 23, 15, 28, 37], [48, 5, 30, 14, 23, 9, 2, 35], [48, 46, 30, 14, 23, 15, 38, 31], [42, 40, 33, 14, 23, 9, 18, 37], [0, 41, 30, 14, 23, 3, 18, 31], [0, 10, 33, 14, 23, 15, 34, 37], [13, 40, 43, 14, 23, 15, 17, 31], [44, 5, 30, 14, 23, 15, 28, 35], [48, 45, 30, 14, 23, 15, 16, 35], [42, 40, 43, 14, 23, 9, 2, 37], [19, 45, 43, 14, 23, 3, 2, 35], [0, 40, 43, 14, 23, 3, 32, 37], [44, 41, 30, 14, 23, 3, 18, 37], [36, 10, 43, 14, 23, 3, 29, 31], [20, 47, 33, 14, 23, 9, 32, 37], [20, 45, 30, 14, 23, 9, 2, 31], [48, 47, 43, 14, 23, 15, 34, 31], [42, 46, 30, 14, 23, 9, 38, 31], [12, 47, 43, 14, 23, 9, 32, 37], [26, 10, 33, 14, 23, 15, 17, 37], [24, 40, 30, 14, 23, 9, 17, 37], [44, 5, 43, 14, 23, 3, 32, 37], [27, 10, 43, 14, 23, 9, 29, 35], [7, 5, 43, 14, 23, 3, 32, 31], [12, 41, 33, 14, 23, 15, 2, 37], [48, 40, 33, 14, 23, 15, 29, 31], [26, 47, 30, 14, 23, 3, 17, 37], [27, 47, 30, 14, 23, 9, 32, 35], [25, 40, 43, 14, 23, 3, 29, 35], [26, 40, 43, 14, 23, 15, 32, 35], [49, 41, 43, 14, 23, 9, 32, 37], [20, 10, 33, 14, 23, 15, 32, 35], [19, 40, 33, 14, 23, 3, 32, 35], [0, 46, 33, 14, 23, 15, 39, 31], [12, 40, 33, 14, 23, 3, 32, 31], [19, 45, 43, 14, 23, 3, 18, 37], [13, 11, 30, 14, 23, 15, 16, 37], [24, 45, 30, 14, 23, 3, 1, 35], [27, 46, 43, 14, 23, 15, 29, 37], [22, 45, 30, 14, 23, 3, 17, 35], [19, 11, 33, 14, 23, 9, 29, 37], [24, 40, 33, 14, 23, 9, 39, 35], [44, 10, 30, 14, 23, 9, 39, 35], [26, 47, 30, 14, 23, 9, 17, 35], [4, 40, 33, 14, 23, 9, 17, 35], [21, 46, 43, 14, 23, 15, 28, 37], [21, 45, 30, 14, 23, 15, 38, 35], [7, 40, 43, 14, 23, 9, 39, 35], [20, 46, 30, 14, 23, 15, 1, 35], [26, 45, 43, 14, 23, 15, 18, 31], [12, 10, 33, 14, 23, 15, 38, 35], [0, 11, 33, 14, 23, 15, 29, 31], [22, 40, 43, 14, 23, 15, 16, 31], [0, 5, 43, 14, 23, 9, 18, 31], [36, 40, 33, 14, 23, 3, 18, 37], [21, 10, 30, 14, 23, 15, 16, 31], [0, 5, 43, 14, 23, 3, 2, 37], [13, 11, 33, 14, 23, 15, 34, 35], [0, 41, 43, 14, 23, 9, 28, 35], [20, 11, 33, 14, 23, 15, 29, 35], [26, 45, 43, 14, 23, 9, 34, 31], [44, 46, 33, 14, 23, 9, 32, 37], [48, 5, 30, 14, 23, 9, 39, 31], [36, 47, 33, 14, 23, 3, 18, 35], [24, 46, 33, 14, 23, 15, 34, 37], [13, 47, 30, 14, 23, 9, 18, 37], [0, 45, 30, 14, 23, 3, 32, 35], [48, 45, 30, 14, 23, 15, 32, 35], [21, 40, 43, 14, 23, 9, 32, 31], [4, 47, 30, 14, 23, 3, 29, 37], [48, 45, 43, 14, 23, 15, 16, 37], [44, 41, 43, 14, 23, 3, 2, 37], [20, 40, 30, 14, 23, 15, 16, 37], [27, 47, 33, 14, 23, 9, 18, 37], [36, 5, 33, 14, 23, 9, 18, 31], [25, 41, 43, 14, 23, 3, 17, 31], [26, 45, 43, 14, 23, 15, 17, 31], [48, 45, 33, 14, 23, 15, 17, 35], [0, 47, 30, 14, 23, 15, 2, 37], [20, 41, 33, 14, 23, 15, 38, 31], [0, 5, 33, 14, 23, 3, 29, 31], [44, 47, 30, 14, 23, 15, 17, 31], [42, 10, 43, 14, 23, 15, 16, 37], [49, 45, 30, 14, 23, 9, 17, 35], [44, 46, 43, 14, 23, 3, 18, 31], [6, 47, 43, 14, 23, 3, 39, 31], [24, 46, 43, 14, 23, 9, 18, 37], [0, 11, 43, 14, 23, 3, 17, 31], [44, 10, 43, 14, 23, 3, 16, 31], [19, 46, 33, 14, 23, 3, 18, 37], [26, 45, 43, 14, 23, 15, 1, 31], [8, 40, 30, 14, 23, 9, 29, 37], [36, 5, 33, 14, 23, 3, 16, 35], [24, 11, 30, 14, 23, 9, 2, 31], [24, 47, 33, 14, 23, 9, 17, 35], [24, 5, 33, 14, 23, 15, 29, 35], [22, 5, 30, 14, 23, 9, 34, 37], [13, 41, 30, 14, 23, 3, 18, 31], [42, 40, 43, 14, 23, 9, 1, 35], [24, 45, 33, 14, 23, 3, 29, 31], [4, 41, 43, 14, 23, 3, 32, 31], [36, 46, 43, 14, 23, 15, 34, 31], [25, 41, 30, 14, 23, 15, 1, 31], [20, 11, 30, 14, 23, 15, 29, 35], [36, 47, 33, 14, 23, 9, 38, 35], [0, 47, 30, 14, 23, 9, 39, 35], [25, 40, 33, 14, 23, 15, 2, 31], [0, 45, 33, 14, 23, 9, 2, 35], [6, 45, 43, 14, 23, 3, 17, 35], [22, 47, 33, 14, 23, 15, 28, 37], [49, 47, 30, 14, 23, 15, 29, 35], [36, 46, 30, 14, 23, 9, 34, 35], [48, 46, 30, 14, 23, 15, 2, 37], [20, 5, 33, 14, 23, 9, 16, 37], [13, 40, 43, 14, 23, 15, 16, 31], [8, 5, 43, 14, 23, 9, 16, 37], [49, 5, 33, 14, 23, 15, 29, 35], [12, 5, 33, 14, 23, 15, 2, 35], [25, 10, 43, 14, 23, 3, 38, 37], [4, 11, 30, 14, 23, 3, 16, 35], [48, 10, 33, 14, 23, 15, 32, 37], [12, 11, 43, 14, 23, 3, 17, 31], [8, 41, 30, 14, 23, 15, 29, 31], [12, 11, 43, 14, 23, 9, 34, 31], [21, 11, 33, 14, 23, 3, 16, 37], [19, 45, 30, 14, 23, 15, 38, 31], [4, 10, 30, 14, 23, 3, 2, 31], [0, 11, 43, 14, 23, 3, 17, 35], [25, 41, 33, 14, 23, 3, 28, 31], [12, 11, 33, 14, 23, 9, 18, 31], [6, 45, 43, 14, 23, 15, 28, 35], [24, 10, 43, 14, 23, 15, 29, 31], [20, 11, 43, 14, 23, 15, 16, 31], [25, 40, 30, 14, 23, 9, 16, 37], [8, 45, 43, 14, 23, 3, 16, 37], [12, 47, 30, 14, 23, 15, 18, 31], [26, 10, 30, 14, 23, 9, 38, 37], [12, 40, 30, 14, 23, 9, 28, 37], [36, 47, 43, 14, 23, 15, 17, 37], [4, 40, 43, 14, 23, 15, 1, 37], [6, 41, 33, 14, 23, 3, 38, 35], [26, 10, 43, 14, 23, 9, 38, 35], [4, 11, 30, 14, 23, 9, 16, 37], [26, 11, 30, 14, 23, 9, 16, 31], [27, 41, 43, 14, 23, 3, 2, 37], [6, 47, 30, 14, 23, 3, 16, 35], [22, 47, 43, 14, 23, 15, 39, 35], [24, 10, 33, 14, 23, 3, 29, 35], [25, 40, 30, 14, 23, 9, 28, 31], [8, 41, 33, 14, 23, 15, 17, 31], [19, 11, 43, 14, 23, 15, 2, 35], [13, 47, 33, 14, 23, 15, 29, 35], [13, 40, 43, 14, 23, 3, 17, 35], [24, 11, 33, 14, 23, 15, 34, 31], [8, 5, 33, 14, 23, 3, 16, 35], [7, 40, 30, 14, 23, 9, 39, 35], [0, 11, 30, 14, 23, 9, 1, 37], [27, 45, 30, 14, 23, 9, 29, 35], [19, 5, 33, 14, 23, 3, 1, 37], [26, 40, 30, 14, 23, 3, 16, 31], [22, 41, 30, 14, 23, 9, 28, 35], [24, 41, 30, 14, 23, 9, 29, 37], [27, 5, 30, 14, 23, 3, 29, 31], [49, 47, 30, 14, 23, 9, 17, 31], [25, 10, 43, 14, 23, 15, 32, 31], [21, 41, 43, 14, 23, 15, 1, 37], [49, 40, 33, 14, 23, 9, 29, 37], [26, 46, 33, 14, 23, 15, 32, 37], [19, 11, 33, 14, 23, 9, 29, 31], [49, 47, 33, 14, 23, 15, 18, 37], [27, 40, 43, 14, 23, 9, 39, 35], [44, 45, 30, 14, 23, 3, 16, 31], [36, 47, 30, 14, 23, 9, 18, 37], [13, 11, 33, 14, 23, 3, 34, 37], [49, 40, 43, 14, 23, 9, 17, 31], [26, 40, 30, 14, 23, 15, 1, 31], [7, 5, 33, 14, 23, 3, 17, 35], [24, 46, 30, 14, 23, 3, 39, 37], [24, 11, 33, 14, 23, 15, 2, 31], [24, 5, 30, 14, 23, 9, 18, 35], [49, 40, 30, 14, 23, 9, 18, 31], [21, 11, 30, 14, 23, 15, 16, 35], [44, 40, 30, 14, 23, 3, 38, 35], [42, 46, 30, 14, 23, 3, 17, 31], [6, 11, 43, 14, 23, 3, 18, 35], [27, 5, 43, 14, 23, 3, 34, 37], [49, 41, 30, 14, 23, 3, 28, 35], [25, 5, 33, 14, 23, 3, 38, 35], [22, 5, 43, 14, 23, 15, 18, 35], [36, 40, 30, 14, 23, 3, 38, 31], [27, 41, 30, 14, 23, 3, 32, 35], [25, 41, 43, 14, 23, 3, 18, 37], [13, 41, 30, 14, 23, 9, 18, 35], [20, 41, 33, 14, 23, 15, 17, 31], [20, 46, 43, 14, 23, 15, 2, 37], [36, 46, 30, 14, 23, 15, 1, 37], [25, 10, 33, 14, 23, 9, 1, 37], [48, 11, 33, 14, 23, 3, 39, 35], [21, 47, 30, 14, 23, 3, 2, 35], [27, 46, 33, 14, 23, 15, 38, 31], [25, 11, 30, 14, 23, 9, 16, 37], [21, 47, 33, 14, 23, 15, 17, 31], [8, 5, 33, 14, 23, 9, 17, 35], [44, 11, 30, 14, 23, 15, 2, 37], [48, 45, 33, 14, 23, 9, 28, 37], [20, 11, 30, 14, 23, 9, 18, 35], [4, 11, 33, 14, 23, 15, 28, 37], [27, 45, 33, 14, 23, 3, 28, 37], [24, 40, 33, 14, 23, 15, 29, 31], [13, 46, 30, 14, 23, 15, 32, 31], [27, 11, 43, 14, 23, 9, 18, 37], [13, 45, 33, 14, 23, 3, 38, 37], [20, 11, 43, 14, 23, 15, 28, 37], [36, 10, 33, 14, 23, 15, 29, 37], [20, 40, 30, 14, 23, 3, 1, 37], [19, 10, 33, 14, 23, 3, 1, 35], [49, 46, 30, 14, 23, 3, 38, 37], [25, 46, 33, 14, 23, 15, 18, 35], [21, 10, 43, 14, 23, 15, 34, 31], [0, 10, 33, 14, 23, 3, 34, 31], [21, 45, 43, 14, 23, 15, 28, 37], [7, 5, 33, 14, 23, 3, 2, 37], [6, 10, 33, 14, 23, 9, 32, 37], [13, 46, 30, 14, 23, 3, 17, 31], [4, 45, 30, 14, 23, 3, 34, 37], [26, 10, 30, 14, 23, 15, 18, 35], [19, 45, 30, 14, 23, 3, 2, 31], [7, 45, 30, 14, 23, 15, 34, 37], [4, 41, 30, 14, 23, 3, 16, 37], [6, 10, 30, 14, 23, 3, 2, 31], [8, 47, 30, 14, 23, 9, 29, 37], [19, 5, 43, 14, 23, 9, 29, 35], [44, 11, 43, 14, 23, 15, 17, 35], [7, 5, 43, 14, 23, 15, 29, 35], [42, 5, 30, 14, 23, 15, 38, 31], [49, 40, 33, 14, 23, 15, 32, 31], [25, 47, 30, 14, 23, 15, 32, 37], [7, 41, 30, 14, 23, 3, 38, 35], [0, 10, 33, 14, 23, 9, 34, 35], [0, 11, 33, 14, 23, 3, 17, 35], [13, 46, 43, 14, 23, 9, 1, 31], [21, 11, 43, 14, 23, 15, 32, 37], [22, 5, 30, 14, 23, 15, 34, 35], [0, 10, 33, 14, 23, 9, 28, 31], [27, 40, 30, 14, 23, 9, 2, 37], [42, 40, 33, 14, 23, 3, 16, 31], [21, 41, 43, 14, 23, 9, 1, 31], [20, 40, 43, 14, 23, 3, 16, 35], [0, 40, 43, 14, 23, 15, 16, 31], [27, 40, 30, 14, 23, 15, 18, 31], [44, 40, 43, 14, 23, 9, 29, 35], [19, 47, 33, 14, 23, 3, 2, 35], [20, 46, 30, 14, 23, 15, 34, 35], [26, 46, 33, 14, 23, 15, 16, 35], [7, 47, 33, 14, 23, 15, 29, 31], [36, 5, 43, 14, 23, 3, 1, 37], [48, 45, 43, 14, 23, 9, 17, 31], [21, 40, 33, 14, 23, 15, 1, 35], [44, 10, 30, 14, 23, 15, 16, 35], [6, 45, 30, 14, 23, 15, 34, 31], [24, 5, 33, 14, 23, 9, 34, 35], [7, 11, 33, 14, 23, 15, 18, 35], [27, 45, 33, 14, 23, 15, 18, 37], [44, 47, 33, 14, 23, 15, 34, 37], [22, 10, 33, 14, 23, 15, 2, 31], [42, 46, 30, 14, 23, 15, 16, 37], [24, 47, 33, 14, 23, 3, 18, 31], [27, 47, 30, 14, 23, 9, 17, 31], [44, 10, 30, 14, 23, 15, 1, 37], [24, 41, 30, 14, 23, 9, 18, 31], [13, 45, 43, 14, 23, 15, 18, 35], [0, 46, 30, 14, 23, 3, 16, 35], [27, 45, 30, 14, 23, 15, 29, 35], [49, 40, 43, 14, 23, 15, 16, 31], [19, 5, 30, 14, 23, 9, 32, 37], [21, 46, 30, 14, 23, 9, 29, 37], [20, 45, 43, 14, 23, 3, 16, 31], [21, 41, 30, 14, 23, 15, 34, 37], [26, 47, 43, 14, 23, 15, 38, 31], [26, 47, 33, 14, 23, 15, 38, 31], [48, 5, 43, 14, 23, 15, 38, 31], [36, 46, 33, 14, 23, 15, 1, 35], [44, 5, 43, 14, 23, 3, 2, 31], [44, 41, 30, 14, 23, 15, 39, 35], [8, 5, 30, 14, 23, 3, 18, 35], [12, 40, 43, 14, 23, 15, 2, 31], [49, 10, 33, 14, 23, 3, 38, 37], [25, 11, 43, 14, 23, 3, 32, 35], [12, 10, 33, 14, 23, 3, 1, 35], [42, 11, 43, 14, 23, 3, 2, 31], [20, 40, 30, 14, 23, 3, 34, 35], [12, 46, 33, 14, 23, 15, 39, 31], [8, 40, 30, 14, 23, 15, 29, 31], [22, 45, 43, 14, 23, 15, 17, 35], [20, 5, 33, 14, 23, 9, 1, 37], [4, 46, 30, 14, 23, 3, 16, 35], [49, 41, 30, 14, 23, 15, 2, 31], [22, 10, 30, 14, 23, 9, 18, 31], [6, 46, 30, 14, 23, 15, 38, 35], [19, 5, 30, 14, 23, 9, 34, 35], [36, 47, 30, 14, 23, 3, 18, 37], [0, 40, 30, 14, 23, 9, 29, 35], [19, 41, 43, 14, 23, 15, 39, 35], [8, 10, 33, 14, 23, 15, 34, 35], [8, 5, 30, 14, 23, 9, 32, 37], [49, 5, 30, 14, 23, 9, 17, 37], [7, 10, 33, 14, 23, 15, 16, 35], [27, 40, 30, 14, 23, 3, 32, 31], [21, 47, 30, 14, 23, 15, 16, 31], [4, 5, 43, 14, 23, 3, 2, 31], [13, 41, 43, 14, 23, 9, 17, 37], [19, 5, 33, 14, 23, 9, 32, 37], [13, 5, 43, 14, 23, 3, 28, 31], [25, 40, 43, 14, 23, 3, 1, 37], [7, 46, 30, 14, 23, 9, 1, 35], [27, 47, 43, 14, 23, 3, 28, 37], [25, 45, 30, 14, 23, 3, 39, 31], [25, 10, 33, 14, 23, 9, 39, 37], [24, 5, 30, 14, 23, 15, 39, 35], [4, 41, 33, 14, 23, 3, 39, 35], [25, 10, 43, 14, 23, 3, 34, 35], [25, 5, 33, 14, 23, 3, 38, 35], [44, 10, 43, 14, 23, 15, 18, 37], [36, 11, 43, 14, 23, 15, 16, 37], [0, 45, 33, 14, 23, 15, 17, 35], [8, 40, 33, 14, 23, 3, 38, 35], [49, 47, 30, 14, 23, 9, 28, 35], [4, 45, 43, 14, 23, 15, 34, 35], [19, 40, 43, 14, 23, 9, 2, 35], [19, 11, 33, 14, 23, 3, 32, 37], [7, 47, 43, 14, 23, 3, 39, 37], [4, 40, 30, 14, 23, 9, 38, 31], [42, 5, 33, 14, 23, 3, 1, 37], [48, 10, 33, 14, 23, 3, 28, 35], [0, 40, 33, 14, 23, 3, 17, 31], [4, 46, 33, 14, 23, 15, 38, 37], [27, 11, 43, 14, 23, 15, 34, 31], [22, 41, 33, 14, 23, 9, 34, 37], [36, 40, 30, 14, 23, 3, 16, 31], [27, 5, 33, 14, 23, 15, 39, 35], [20, 5, 43, 14, 23, 3, 2, 37], [0, 45, 30, 14, 23, 3, 18, 31], [20, 11, 33, 14, 23, 9, 16, 35], [4, 46, 43, 14, 23, 3, 34, 35], [42, 45, 33, 14, 23, 15, 18, 31], [22, 46, 33, 14, 23, 9, 34, 31], [20, 45, 30, 14, 23, 15, 39, 35], [13, 10, 33, 14, 23, 3, 17, 37], [13, 47, 43, 14, 23, 3, 32, 31], [25, 45, 30, 14, 23, 9, 16, 37], [22, 46, 43, 14, 23, 9, 29, 37], [44, 40, 30, 14, 23, 3, 39, 31], [26, 40, 33, 14, 23, 9, 2, 37], [7, 46, 33, 14, 23, 9, 32, 37], [6, 5, 33, 14, 23, 15, 2, 35], [26, 5, 30, 14, 23, 3, 18, 37], [27, 46, 33, 14, 23, 15, 1, 35], [13, 40, 33, 14, 23, 9, 29, 37], [36, 10, 30, 14, 23, 9, 29, 37], [4, 47, 30, 14, 23, 3, 2, 31], [21, 10, 30, 14, 23, 9, 2, 31], [36, 10, 33, 14, 23, 3, 17, 37], [36, 40, 33, 14, 23, 9, 38, 31], [19, 10, 33, 14, 23, 3, 16, 31], [7, 5, 43, 14, 23, 15, 18, 37], [19, 46, 33, 14, 23, 15, 2, 35], [42, 40, 33, 14, 23, 15, 2, 31], [26, 5, 30, 14, 23, 3, 34, 35], [49, 40, 33, 14, 23, 9, 29, 37], [12, 45, 43, 14, 23, 3, 1, 37], [20, 41, 30, 14, 23, 15, 2, 35], [7, 47, 30, 14, 23, 9, 1, 31], [13, 41, 30, 14, 23, 15, 38, 35], [12, 5, 43, 14, 23, 9, 2, 35], [27, 41, 30, 14, 23, 9, 39, 35], [19, 46, 30, 14, 23, 15, 38, 37], [26, 11, 33, 14, 23, 15, 29, 35], [20, 46, 30, 14, 23, 3, 32, 35], [25, 47, 33, 14, 23, 15, 18, 37], [7, 5, 30, 14, 23, 3, 1, 37], [6, 46, 30, 14, 23, 15, 16, 37], [20, 46, 43, 14, 23, 9, 18, 37], [26, 46, 43, 14, 23, 3, 38, 31], [8, 5, 43, 14, 23, 9, 32, 35], [8, 40, 43, 14, 23, 3, 18, 31], [27, 11, 33, 14, 23, 3, 39, 31], [27, 10, 30, 14, 23, 3, 32, 35], [19, 47, 33, 14, 23, 3, 32, 35], [44, 10, 30, 14, 23, 3, 32, 35], [26, 46, 33, 14, 23, 9, 32, 35], [13, 11, 43, 14, 23, 15, 17, 31], [24, 46, 30, 14, 23, 3, 34, 31], [24, 45, 43, 14, 23, 15, 1, 35], [42, 47, 43, 14, 23, 15, 16, 31]]
        if how_many_parts == 10:
            picked_model = [[26, 45, 30, 14, 23], [3, 18, 37, 44, 5], [30, 14, 23, 3, 18], [35, 49, 46, 30, 14], [23, 9, 17, 35, 0], [46, 33, 14, 23, 15], [28, 31, 0, 46, 33], [14, 23, 9, 32, 31], [13, 46, 30, 14, 23], [3, 18, 31, 27, 5], [30, 14, 23, 3, 17], [37, 22, 11, 30, 14], [23, 15, 2, 31, 19], [11, 30, 14, 23, 15], [2, 31, 25, 41, 30], [14, 23, 15, 16, 37], [21, 10, 33, 14, 23], [3, 1, 37, 44, 10], [33, 14, 23, 3, 38], [35, 26, 41, 30, 14], [23, 3, 18, 37, 27], [47, 33, 14, 23, 15], [28, 35, 7, 40, 43], [14, 23, 15, 18, 35], [48, 41, 30, 14, 23], [3, 18, 37, 12, 46], [33, 14, 23, 3, 16], [37, 25, 11, 43, 14], [23, 3, 1, 35, 22], [47, 43, 14, 23, 15], [2, 37, 25, 10, 30], [14, 23, 3, 28, 35], [27, 40, 43, 14, 23], [15, 39, 37, 8, 11], [30, 14, 23, 15, 29], [35, 25, 47, 30, 14], [23, 9, 18, 35, 49], [45, 43, 14, 23, 15], [17, 31, 49, 40, 30], [14, 23, 15, 38, 31], [25, 46, 30, 14, 23], [9, 1, 31, 12, 40], [30, 14, 23, 3, 1], [35, 44, 41, 43, 14], [23, 3, 29, 35, 26], [5, 43, 14, 23, 9], [2, 31, 49, 45, 30], [14, 23, 9, 1, 37], [7, 47, 43, 14, 23], [9, 17, 35, 22, 11], [33, 14, 23, 3, 32], [35, 4, 5, 43, 14], [23, 9, 34, 35, 26], [41, 30, 14, 23, 9], [39, 37, 24, 45, 43], [14, 23, 15, 39, 31], [42, 45, 43, 14, 23], [15, 1, 37, 24, 10], [43, 14, 23, 3, 38], [31, 36, 41, 30, 14], [23, 15, 2, 37, 21], [46, 33, 14, 23, 9], [18, 37, 49, 40, 30], [14, 23, 3, 1, 35], [44, 47, 30, 14, 23], [15, 2, 35, 12, 47], [30, 14, 23, 15, 38], [37, 19, 11, 30, 14], [23, 3, 29, 35, 42], [47, 30, 14, 23, 15], [39, 35, 6, 41, 43], [14, 23, 9, 18, 37], [7, 11, 30, 14, 23], [15, 29, 31, 21, 40], [30, 14, 23, 9, 32], [37, 36, 41, 30, 14], [23, 3, 28, 37, 0], [41, 30, 14, 23, 9], [2, 37, 0, 10, 43], [14, 23, 15, 39, 37], [4, 5, 30, 14, 23], [15, 1, 31, 0, 46], [43, 14, 23, 3, 34], [35, 6, 40, 33, 14], [23, 15, 28, 37, 8], [46, 43, 14, 23, 15], [2, 37, 36, 5, 30], [14, 23, 3, 32, 37], [19, 46, 30, 14, 23], [3, 38, 35, 19, 46], [33, 14, 23, 3, 39], [37, 25, 45, 43, 14], [23, 15, 17, 31, 24], [11, 30, 14, 23, 3], [17, 37, 8, 46, 30], [14, 23, 3, 39, 37], [8, 40, 33, 14, 23], [15, 18, 31, 12, 47], [43, 14, 23, 3, 38], [31, 6, 11, 30, 14], [23, 9, 39, 37, 21], [41, 30, 14, 23, 9], [1, 31, 24, 46, 43], [14, 23, 3, 29, 37], [42, 11, 43, 14, 23], [9, 2, 31, 22, 11], [33, 14, 23, 15, 34], [37, 26, 11, 33, 14], [23, 3, 18, 31, 12], [41, 33, 14, 23, 9], [32, 31, 7, 40, 30], [14, 23, 15, 2, 37], [24, 46, 30, 14, 23], [3, 39, 31, 26, 40], [43, 14, 23, 3, 28], [35, 7, 40, 43, 14], [23, 15, 39, 35, 0], [41, 33, 14, 23, 15], [32, 31, 49, 45, 30], [14, 23, 9, 18, 31], [4, 5, 30, 14, 23], [15, 28, 31, 42, 11], [33, 14, 23, 9, 2], [31, 49, 10, 30, 14], [23, 9, 16, 37, 27], [11, 30, 14, 23, 15], [38, 35, 27, 11, 33], [14, 23, 3, 39, 31], [13, 47, 33, 14, 23], [9, 1, 31, 0, 46], [33, 14, 23, 9, 34], [31, 0, 46, 30, 14], [23, 3, 16, 37, 49], [11, 30, 14, 23, 3], [34, 31, 7, 10, 30], [14, 23, 9, 1, 35], [12, 46, 33, 14, 23], [9, 29, 35, 7, 47], [33, 14, 23, 15, 29], [35, 24, 41, 30, 14], [23, 3, 16, 35, 44], [40, 30, 14, 23, 15], [38, 35, 6, 46, 33], [14, 23, 3, 32, 37], [24, 10, 33, 14, 23], [9, 39, 35, 19, 47], [33, 14, 23, 9, 28], [35, 48, 11, 30, 14], [23, 3, 32, 31, 19], [11, 30, 14, 23, 3], [17, 35, 19, 11, 30], [14, 23, 9, 2, 37], [36, 11, 43, 14, 23], [9, 39, 31, 6, 46], [30, 14, 23, 9, 17], [31, 49, 45, 43, 14], [23, 9, 18, 35, 36], [47, 30, 14, 23, 3], [29, 31, 49, 5, 33], [14, 23, 3, 34, 31], [0, 5, 30, 14, 23], [15, 39, 35, 25, 45], [33, 14, 23, 3, 18], [35, 36, 11, 33, 14], [23, 15, 32, 31, 26], [11, 43, 14, 23, 9], [1, 35, 21, 46, 33], [14, 23, 9, 34, 31], [26, 40, 33, 14, 23], [3, 38, 37, 6, 41], [43, 14, 23, 9, 1], [37, 44, 41, 30, 14], [23, 3, 29, 35, 7], [11, 43, 14, 23, 3], [2, 31, 48, 11, 30], [14, 23, 9, 32, 31], [0, 41, 43, 14, 23], [3, 29, 37, 26, 45], [33, 14, 23, 9, 1], [35, 44, 46, 43, 14], [23, 3, 17, 37, 7], [41, 33, 14, 23, 3], [28, 31, 6, 10, 33], [14, 23, 15, 2, 35], [26, 10, 33, 14, 23], [3, 29, 31, 4, 5], [30, 14, 23, 3, 38], [31, 36, 45, 43, 14], [23, 3, 2, 37, 25], [41, 43, 14, 23, 15], [38, 37, 42, 46, 30], [14, 23, 15, 32, 31], [44, 40, 43, 14, 23], [9, 38, 35, 12, 5], [43, 14, 23, 9, 32], [31, 48, 41, 30, 14], [23, 3, 17, 31, 26], [41, 33, 14, 23, 3], [16, 35, 8, 47, 30], [14, 23, 9, 1, 37], [4, 11, 33, 14, 23], [3, 16, 37, 44, 41], [33, 14, 23, 9, 2], [37, 19, 45, 30, 14], [23, 9, 28, 35, 48], [46, 33, 14, 23, 9], [32, 35, 0, 5, 33], [14, 23, 9, 2, 35], [20, 47, 30, 14, 23], [9, 29, 37, 25, 40], [30, 14, 23, 9, 28], [31, 42, 41, 30, 14], [23, 9, 32, 35, 8], [41, 43, 14, 23, 15], [1, 37, 27, 10, 33], [14, 23, 3, 38, 35], [36, 40, 43, 14, 23], [9, 29, 37, 27, 10], [30, 14, 23, 3, 29], [31, 12, 40, 43, 14], [23, 9, 32, 35, 8], [45, 33, 14, 23, 9], [39, 37, 0, 5, 33], [14, 23, 9, 38, 37], [6, 5, 33, 14, 23], [9, 18, 35, 44, 40], [43, 14, 23, 15, 28], [35, 0, 46, 43, 14], [23, 15, 2, 31, 4], [41, 33, 14, 23, 15], [29, 35, 7, 5, 30], [14, 23, 3, 32, 31], [19, 40, 33, 14, 23], [3, 39, 31, 4, 47], [30, 14, 23, 3, 32], [31, 25, 40, 33, 14], [23, 9, 18, 31, 7], [10, 30, 14, 23, 3], [18, 31, 26, 47, 33], [14, 23, 9, 17, 31], [36, 46, 43, 14, 23], [15, 39, 31, 49, 41], [30, 14, 23, 9, 1], [37, 7, 10, 43, 14], [23, 3, 28, 31, 6], [47, 30, 14, 23, 9], [28, 37, 42, 46, 33], [14, 23, 15, 28, 35], [27, 47, 30, 14, 23], [9, 18, 35, 22, 46], [33, 14, 23, 15, 29], [35, 7, 10, 43, 14], [23, 3, 28, 37, 7], [45, 30, 14, 23, 9], [17, 35, 42, 5, 33], [14, 23, 15, 38, 31], [21, 5, 43, 14, 23], [3, 17, 37, 25, 47], [30, 14, 23, 3, 28], [31, 4, 40, 30, 14], [23, 9, 28, 35, 22], [5, 30, 14, 23, 9], [34, 35, 22, 11, 30], [14, 23, 9, 32, 31], [42, 45, 30, 14, 23], [3, 17, 35, 0, 40], [33, 14, 23, 9, 39], [37, 42, 41, 43, 14], [23, 15, 32, 37, 0], [46, 43, 14, 23, 9], [32, 37, 19, 41, 33], [14, 23, 3, 2, 35], [8, 40, 30, 14, 23], [15, 2, 37, 44, 10], [43, 14, 23, 9, 34], [35, 21, 11, 43, 14], [23, 9, 18, 35, 42], [41, 33, 14, 23, 9], [39, 35, 12, 41, 43], [14, 23, 15, 17, 31], [36, 10, 33, 14, 23], [3, 28, 31, 4, 47], [30, 14, 23, 3, 2], [31, 7, 47, 33, 14], [23, 3, 1, 31, 44], [46, 30, 14, 23, 15], [34, 31, 44, 40, 43], [14, 23, 15, 38, 31], [27, 40, 33, 14, 23], [9, 38, 35, 4, 41], [33, 14, 23, 15, 28], [35, 13, 46, 30, 14], [23, 15, 38, 35, 21], [46, 33, 14, 23, 15], [1, 37, 12, 40, 43], [14, 23, 15, 34, 31], [12, 46, 30, 14, 23], [3, 1, 37, 25, 40], [43, 14, 23, 3, 18], [37, 19, 46, 43, 14], [23, 9, 16, 31, 48], [5, 30, 14, 23, 3], [2, 35, 6, 41, 30], [14, 23, 3, 2, 31], [49, 45, 43, 14, 23], [15, 28, 35, 24, 41], [30, 14, 23, 15, 34], [37, 44, 5, 30, 14], [23, 9, 34, 35, 6], [40, 30, 14, 23, 3], [38, 31, 21, 10, 43], [14, 23, 3, 17, 37], [4, 11, 43, 14, 23], [3, 38, 35, 22, 40], [33, 14, 23, 15, 32], [35, 22, 45, 30, 14], [23, 9, 38, 35, 36], [40, 30, 14, 23, 9], [16, 37, 44, 10, 43], [14, 23, 3, 32, 37], [21, 46, 30, 14, 23], [3, 32, 31, 26, 46], [43, 14, 23, 3, 16], [35, 19, 10, 33, 14], [23, 3, 32, 31, 24], [40, 30, 14, 23, 3], [29, 31, 42, 40, 33], [14, 23, 15, 39, 31], [42, 45, 33, 14, 23], [3, 39, 37, 24, 46], [30, 14, 23, 3, 17], [31, 48, 46, 43, 14], [23, 3, 18, 35, 25], [40, 43, 14, 23, 15], [18, 31, 12, 46, 33], [14, 23, 9, 18, 31], [0, 40, 33, 14, 23], [15, 18, 37, 8, 47], [43, 14, 23, 15, 32], [37, 36, 45, 33, 14], [23, 3, 34, 37, 13], [46, 43, 14, 23, 3], [32, 31, 26, 41, 30], [14, 23, 9, 16, 35], [49, 5, 30, 14, 23], [3, 39, 31, 4, 11], [30, 14, 23, 15, 2], [31, 48, 5, 33, 14], [23, 9, 1, 35, 44], [10, 43, 14, 23, 3], [1, 35, 8, 45, 43], [14, 23, 3, 34, 31], [49, 11, 43, 14, 23], [9, 17, 37, 44, 10], [33, 14, 23, 9, 29], [31, 27, 10, 43, 14], [23, 15, 17, 31, 8], [45, 30, 14, 23, 3], [16, 37, 27, 45, 30], [14, 23, 3, 29, 35], [0, 40, 30, 14, 23], [9, 1, 31, 0, 41], [30, 14, 23, 15, 2], [37, 0, 47, 30, 14], [23, 9, 2, 37, 26], [41, 33, 14, 23, 15], [39, 31, 19, 46, 43], [14, 23, 3, 29, 31], [12, 45, 30, 14, 23], [3, 16, 37, 36, 46], [43, 14, 23, 9, 32], [37, 19, 5, 30, 14], [23, 3, 38, 37, 6], [5, 30, 14, 23, 3], [28, 31, 26, 47, 30], [14, 23, 15, 17, 31], [19, 10, 43, 14, 23], [15, 18, 31, 7, 47], [43, 14, 23, 9, 32], [35, 13, 47, 33, 14], [23, 3, 39, 31, 8], [41, 43, 14, 23, 9], [17, 37, 21, 40, 33], [14, 23, 3, 1, 35], [4, 40, 43, 14, 23], [3, 1, 37, 26, 10], [43, 14, 23, 9, 1], [37, 13, 47, 30, 14], [23, 15, 32, 31, 12], [40, 43, 14, 23, 9], [32, 35, 26, 40, 33], [14, 23, 15, 18, 37], [49, 5, 43, 14, 23], [3, 28, 37, 36, 41], [33, 14, 23, 15, 38], [37, 20, 10, 43, 14], [23, 9, 17, 35, 22], [46, 30, 14, 23, 9], [16, 31, 42, 11, 33], [14, 23, 3, 39, 31]]
        if how_many_parts == 20:
            picked_model = [[20, 47, 43, 14, 23, 15, 1, 37, 22, 41], [30, 14, 23, 15, 29, 31, 7, 40, 30, 14], [23, 9, 28, 37, 7, 45, 30, 14, 23, 15], [28, 35, 22, 46, 33, 14, 23, 9, 34, 31], [27, 47, 30, 14, 23, 3, 18, 31, 6, 41], [33, 14, 23, 3, 32, 31, 4, 11, 30, 14], [23, 15, 38, 37, 26, 5, 30, 14, 23, 3], [16, 35, 49, 41, 33, 14, 23, 15, 39, 31], [49, 45, 43, 14, 23, 9, 18, 37, 49, 10], [43, 14, 23, 9, 17, 31, 22, 5, 30, 14], [23, 3, 16, 31, 4, 5, 43, 14, 23, 3], [29, 37, 42, 47, 43, 14, 23, 9, 17, 31], [19, 47, 30, 14, 23, 3, 39, 35, 42, 10], [33, 14, 23, 9, 34, 35, 22, 46, 30, 14], [23, 9, 38, 35, 13, 47, 33, 14, 23, 9], [2, 35, 12, 40, 43, 14, 23, 15, 18, 35], [49, 41, 43, 14, 23, 3, 28, 31, 4, 10], [33, 14, 23, 3, 39, 37, 48, 10, 33, 14], [23, 15, 2, 31, 19, 10, 33, 14, 23, 9], [32, 35, 48, 5, 33, 14, 23, 9, 39, 31], [24, 10, 30, 14, 23, 3, 34, 35, 27, 45], [33, 14, 23, 15, 17, 31, 4, 11, 30, 14], [23, 3, 29, 37, 12, 5, 33, 14, 23, 3], [29, 35, 22, 47, 30, 14, 23, 3, 2, 35], [4, 41, 43, 14, 23, 15, 39, 35, 21, 11], [33, 14, 23, 15, 32, 31, 25, 45, 30, 14], [23, 3, 29, 35, 4, 5, 30, 14, 23, 15], [29, 35, 4, 45, 30, 14, 23, 15, 39, 35], [20, 45, 30, 14, 23, 3, 18, 35, 20, 46], [30, 14, 23, 3, 16, 37, 49, 45, 43, 14], [23, 3, 18, 37, 13, 45, 33, 14, 23, 15], [29, 35, 7, 47, 43, 14, 23, 3, 2, 37], [19, 41, 33, 14, 23, 15, 39, 35, 0, 5], [30, 14, 23, 15, 38, 37, 25, 11, 30, 14], [23, 3, 29, 35, 8, 5, 30, 14, 23, 15], [1, 37, 24, 45, 30, 14, 23, 15, 32, 31], [12, 10, 33, 14, 23, 3, 28, 31, 7, 40], [43, 14, 23, 3, 17, 37, 7, 11, 30, 14], [23, 9, 34, 31, 21, 11, 43, 14, 23, 3], [2, 37, 21, 46, 43, 14, 23, 15, 34, 31], [27, 45, 33, 14, 23, 15, 17, 31, 13, 40], [30, 14, 23, 3, 1, 35, 13, 47, 30, 14], [23, 3, 18, 31, 20, 47, 43, 14, 23, 9], [34, 35, 4, 45, 33, 14, 23, 15, 2, 37], [42, 40, 30, 14, 23, 9, 2, 35, 6, 47], [43, 14, 23, 3, 38, 37, 26, 40, 43, 14], [23, 3, 28, 31, 48, 10, 30, 14, 23, 15], [1, 37, 48, 45, 33, 14, 23, 3, 17, 35], [26, 5, 30, 14, 23, 9, 32, 37, 7, 46], [33, 14, 23, 3, 34, 31, 7, 11, 33, 14], [23, 9, 16, 37, 12, 47, 43, 14, 23, 15], [32, 37, 0, 41, 33, 14, 23, 3, 18, 31], [7, 47, 43, 14, 23, 9, 16, 31, 6, 40], [33, 14, 23, 15, 18, 37, 36, 11, 30, 14], [23, 9, 18, 37, 22, 47, 43, 14, 23, 3], [39, 35, 7, 47, 30, 14, 23, 3, 17, 35], [20, 41, 43, 14, 23, 9, 34, 37, 42, 5], [30, 14, 23, 15, 1, 35, 36, 47, 33, 14], [23, 15, 32, 37, 49, 40, 30, 14, 23, 15], [29, 35, 7, 40, 43, 14, 23, 15, 29, 31], [42, 10, 30, 14, 23, 15, 2, 31, 7, 40], [43, 14, 23, 15, 39, 31, 22, 45, 33, 14], [23, 3, 38, 35, 13, 11, 33, 14, 23, 3], [32, 31, 24, 5, 30, 14, 23, 3, 1, 35], [13, 46, 43, 14, 23, 9, 1, 35, 13, 47], [43, 14, 23, 15, 39, 31, 22, 41, 33, 14], [23, 3, 1, 35, 0, 5, 30, 14, 23, 3], [2, 37, 49, 45, 33, 14, 23, 3, 16, 35], [8, 41, 30, 14, 23, 15, 17, 35, 8, 11], [30, 14, 23, 3, 2, 37, 22, 40, 43, 14], [23, 9, 39, 35, 44, 5, 43, 14, 23, 15], [34, 37, 36, 5, 30, 14, 23, 15, 29, 35], [22, 5, 30, 14, 23, 15, 39, 31, 27, 46], [30, 14, 23, 9, 32, 37, 0, 46, 33, 14], [23, 3, 32, 31, 26, 5, 33, 14, 23, 9], [2, 35, 36, 40, 30, 14, 23, 9, 32, 31], [7, 10, 33, 14, 23, 3, 16, 35, 22, 10], [33, 14, 23, 9, 38, 35, 42, 41, 43, 14], [23, 3, 1, 37, 6, 11, 43, 14, 23, 15], [18, 35, 20, 47, 43, 14, 23, 15, 34, 37], [21, 41, 33, 14, 23, 9, 38, 35, 48, 46], [43, 14, 23, 15, 1, 35, 7, 41, 30, 14], [23, 3, 28, 31, 7, 41, 33, 14, 23, 3], [18, 31, 6, 5, 43, 14, 23, 3, 28, 37], [13, 41, 33, 14, 23, 3, 28, 37, 22, 40], [33, 14, 23, 3, 29, 31, 19, 10, 43, 14], [23, 3, 16, 35, 22, 47, 30, 14, 23, 9], [16, 35, 13, 41, 30, 14, 23, 3, 38, 37], [8, 45, 30, 14, 23, 15, 38, 31, 36, 10], [30, 14, 23, 3, 28, 37, 26, 40, 33, 14], [23, 3, 32, 31, 21, 11, 43, 14, 23, 3], [2, 31, 22, 40, 33, 14, 23, 15, 2, 35], [20, 46, 33, 14, 23, 9, 2, 35, 12, 11], [33, 14, 23, 9, 29, 31, 49, 11, 33, 14], [23, 15, 28, 35, 44, 46, 43, 14, 23, 9], [28, 35, 24, 47, 30, 14, 23, 9, 1, 31], [44, 11, 30, 14, 23, 3, 18, 35, 13, 40], [43, 14, 23, 15, 39, 37, 36, 10, 33, 14], [23, 15, 18, 35, 20, 47, 33, 14, 23, 3], [1, 35, 21, 11, 30, 14, 23, 9, 2, 37], [21, 41, 43, 14, 23, 9, 38, 35, 22, 40], [30, 14, 23, 15, 28, 31, 25, 45, 30, 14], [23, 15, 28, 31, 19, 47, 33, 14, 23, 15], [17, 37, 4, 40, 43, 14, 23, 15, 32, 35], [48, 46, 43, 14, 23, 3, 28, 35, 12, 46], [43, 14, 23, 3, 28, 35, 8, 11, 33, 14], [23, 9, 16, 35, 42, 47, 30, 14, 23, 3], [17, 37, 0, 10, 30, 14, 23, 15, 17, 31], [42, 10, 30, 14, 23, 3, 39, 37, 19, 46], [33, 14, 23, 3, 34, 37, 6, 47, 43, 14], [23, 9, 17, 35, 44, 46, 43, 14, 23, 9], [34, 35, 21, 10, 30, 14, 23, 9, 39, 31], [21, 40, 43, 14, 23, 15, 38, 31, 7, 10], [33, 14, 23, 15, 18, 37, 44, 10, 33, 14], [23, 15, 17, 35, 7, 45, 43, 14, 23, 3], [16, 35, 13, 41, 43, 14, 23, 15, 17, 37], [24, 47, 33, 14, 23, 3, 17, 37, 21, 40], [30, 14, 23, 15, 17, 37, 48, 41, 33, 14], [23, 15, 39, 35, 20, 45, 30, 14, 23, 15], [28, 37, 4, 46, 33, 14, 23, 3, 39, 31], [44, 41, 33, 14, 23, 15, 28, 37, 36, 40], [43, 14, 23, 3, 38, 37, 49, 41, 30, 14], [23, 3, 17, 37, 44, 41, 30, 14, 23, 15], [34, 31, 24, 5, 30, 14, 23, 15, 38, 35], [19, 11, 43, 14, 23, 9, 32, 31, 8, 45], [30, 14, 23, 9, 16, 31, 13, 47, 30, 14], [23, 15, 18, 35, 26, 46, 43, 14, 23, 15], [29, 35, 6, 45, 30, 14, 23, 9, 34, 37], [24, 40, 30, 14, 23, 9, 1, 35, 44, 45], [30, 14, 23, 3, 34, 35, 19, 40, 33, 14], [23, 9, 28, 35, 42, 10, 30, 14, 23, 9], [1, 31, 25, 10, 30, 14, 23, 9, 32, 35], [6, 41, 43, 14, 23, 3, 38, 31, 44, 40], [30, 14, 23, 15, 34, 35, 42, 47, 43, 14], [23, 3, 39, 37, 27, 45, 43, 14, 23, 9], [18, 37, 12, 10, 43, 14, 23, 9, 1, 35], [6, 40, 43, 14, 23, 9, 18, 37, 48, 11], [30, 14, 23, 15, 1, 35, 4, 46, 43, 14], [23, 3, 29, 35, 25, 10, 43, 14, 23, 15], [2, 35, 12, 41, 43, 14, 23, 9, 28, 31], [7, 10, 30, 14, 23, 3, 28, 37, 25, 10], [33, 14, 23, 9, 34, 35, 49, 11, 43, 14], [23, 15, 1, 37, 12, 46, 30, 14, 23, 15], [29, 31, 36, 5, 33, 14, 23, 9, 32, 31], [12, 45, 33, 14, 23, 3, 16, 37, 4, 11], [33, 14, 23, 9, 2, 31, 0, 46, 30, 14], [23, 15, 32, 37, 20, 41, 43, 14, 23, 3], [1, 31, 25, 40, 33, 14, 23, 15, 1, 31], [20, 5, 30, 14, 23, 9, 32, 37, 22, 5], [33, 14, 23, 9, 17, 35, 22, 5, 30, 14], [23, 9, 39, 37, 27, 41, 43, 14, 23, 9], [32, 35, 13, 10, 43, 14, 23, 15, 32, 37], [36, 10, 30, 14, 23, 15, 29, 31, 0, 45], [30, 14, 23, 9, 16, 35, 12, 40, 43, 14], [23, 9, 29, 35, 42, 47, 43, 14, 23, 15], [34, 35, 25, 11, 30, 14, 23, 15, 16, 35], [44, 45, 43, 14, 23, 9, 32, 35, 49, 40], [33, 14, 23, 9, 16, 35, 25, 11, 30, 14], [23, 9, 18, 37, 7, 11, 43, 14, 23, 3], [32, 35, 24, 5, 43, 14, 23, 9, 2, 31], [36, 47, 43, 14, 23, 9, 2, 35, 19, 45], [30, 14, 23, 9, 29, 35, 20, 5, 43, 14], [23, 15, 16, 31, 8, 41, 30, 14, 23, 15], [18, 37, 36, 41, 33, 14, 23, 9, 34, 35], [44, 10, 33, 14, 23, 9, 18, 31, 8, 41], [30, 14, 23, 3, 39, 35, 21, 5, 30, 14], [23, 9, 16, 35, 22, 10, 43, 14, 23, 3], [2, 31, 13, 10, 33, 14, 23, 3, 32, 37], [27, 11, 30, 14, 23, 3, 32, 37, 36, 41], [33, 14, 23, 15, 16, 37, 8, 10, 30, 14], [23, 15, 2, 37, 8, 41, 30, 14, 23, 15], [29, 35, 7, 46, 33, 14, 23, 15, 32, 35], [7, 47, 30, 14, 23, 3, 16, 31, 6, 5], [30, 14, 23, 15, 32, 35, 24, 5, 30, 14], [23, 9, 29, 31, 26, 40, 30, 14, 23, 9], [34, 31, 19, 41, 43, 14, 23, 9, 28, 37], [20, 47, 43, 14, 23, 15, 32, 37, 27, 45], [30, 14, 23, 9, 32, 31, 36, 46, 30, 14], [23, 3, 39, 31, 12, 11, 33, 14, 23, 15], [1, 37, 44, 11, 43, 14, 23, 9, 38, 35], [42, 11, 33, 14, 23, 15, 17, 35, 13, 47], [30, 14, 23, 15, 18, 31, 7, 11, 33, 14], [23, 9, 38, 31, 27, 45, 43, 14, 23, 3], [28, 37, 4, 41, 30, 14, 23, 15, 28, 31], [27, 41, 33, 14, 23, 9, 34, 37, 21, 45], [30, 14, 23, 9, 38, 37, 27, 11, 43, 14], [23, 9, 17, 35, 25, 5, 30, 14, 23, 3], [29, 31, 6, 41, 30, 14, 23, 3, 34, 37], [24, 41, 43, 14, 23, 15, 39, 31, 48, 41], [30, 14, 23, 3, 16, 31, 26, 40, 43, 14], [23, 9, 2, 35, 12, 40, 43, 14, 23, 15], [18, 31, 44, 41, 33, 14, 23, 9, 17, 37], [42, 41, 33, 14, 23, 15, 1, 35, 26, 5], [33, 14, 23, 9, 39, 37, 4, 45, 43, 14], [23, 15, 17, 37, 7, 11, 30, 14, 23, 3], [32, 35, 48, 45, 43, 14, 23, 9, 2, 31], [25, 40, 43, 14, 23, 15, 16, 35, 7, 45], [43, 14, 23, 15, 39, 35, 7, 5, 43, 14], [23, 3, 38, 37, 22, 41, 30, 14, 23, 15], [18, 35, 44, 41, 33, 14, 23, 3, 16, 31], [4, 45, 30, 14, 23, 3, 17, 37, 20, 40], [33, 14, 23, 15, 2, 31, 19, 11, 33, 14], [23, 15, 1, 35, 26, 41, 30, 14, 23, 3], [29, 37, 6, 41, 30, 14, 23, 15, 1, 31], [7, 45, 30, 14, 23, 3, 2, 31, 13, 46], [33, 14, 23, 3, 17, 31, 4, 40, 30, 14], [23, 9, 16, 31, 20, 40, 43, 14, 23, 3], [2, 37, 21, 41, 30, 14, 23, 9, 32, 37], [22, 40, 30, 14, 23, 3, 28, 37, 6, 10], [33, 14, 23, 3, 16, 31, 26, 11, 33, 14], [23, 9, 1, 31, 25, 40, 33, 14, 23, 15], [28, 35, 4, 46, 30, 14, 23, 9, 16, 35], [24, 5, 33, 14, 23, 15, 18, 31, 6, 11], [43, 14, 23, 9, 17, 37, 13, 46, 30, 14], [23, 15, 39, 35, 0, 11, 43, 14, 23, 15], [17, 31, 19, 46, 33, 14, 23, 9, 32, 35], [20, 10, 33, 14, 23, 3, 32, 35, 20, 11], [43, 14, 23, 9, 28, 31, 27, 11, 33, 14], [23, 9, 29, 31, 27, 40, 33, 14, 23, 9], [39, 35, 7, 10, 30, 14, 23, 3, 28, 37], [48, 11, 43, 14, 23, 15, 18, 31, 4, 11], [30, 14, 23, 15, 38, 31, 20, 41, 43, 14], [23, 3, 1, 31, 27, 11, 30, 14, 23, 9], [38, 35, 7, 40, 43, 14, 23, 15, 17, 31], [8, 45, 43, 14, 23, 3, 39, 31, 26, 11], [43, 14, 23, 9, 2, 35, 13, 47, 43, 14], [23, 3, 17, 31, 21, 40, 33, 14, 23, 15], [2, 37, 20, 11, 30, 14, 23, 9, 16, 37], [22, 40, 33, 14, 23, 15, 18, 37, 20, 41], [33, 14, 23, 3, 1, 37, 26, 5, 33, 14], [23, 15, 1, 37, 7, 41, 30, 14, 23, 15], [28, 35, 20, 45, 30, 14, 23, 15, 1, 35], [26, 10, 30, 14, 23, 9, 28, 37, 8, 41], [33, 14, 23, 15, 29, 31, 4, 46, 30, 14], [23, 9, 18, 37, 36, 46, 43, 14, 23, 15], [17, 37, 13, 5, 33, 14, 23, 15, 17, 35], [24, 46, 30, 14, 23, 15, 32, 31, 36, 41], [43, 14, 23, 3, 16, 35, 12, 46, 33, 14], [23, 3, 34, 35, 7, 41, 43, 14, 23, 9], [38, 37, 48, 40, 30, 14, 23, 15, 1, 31], [4, 46, 43, 14, 23, 9, 1, 37, 26, 41], [33, 14, 23, 9, 17, 31, 36, 41, 30, 14], [23, 3, 39, 37, 4, 46, 30, 14, 23, 3], [18, 37, 6, 11, 30, 14, 23, 15, 16, 37], [7, 11, 33, 14, 23, 15, 29, 35, 44, 40], [43, 14, 23, 15, 1, 31, 0, 41, 30, 14], [23, 15, 16, 31, 22, 45, 43, 14, 23, 3], [38, 35, 22, 11, 43, 14, 23, 9, 29, 37], [49, 11, 33, 14, 23, 3, 29, 31, 8, 45], [30, 14, 23, 9, 32, 37, 21, 47, 30, 14], [23, 15, 32, 35, 4, 10, 43, 14, 23, 3], [18, 37, 49, 11, 43, 14, 23, 9, 17, 31], [27, 41, 43, 14, 23, 15, 38, 35, 13, 46], [30, 14, 23, 15, 16, 31, 20, 11, 43, 14], [23, 15, 17, 31, 24, 46, 30, 14, 23, 9], [39, 37, 24, 5, 33, 14, 23, 15, 34, 35], [21, 40, 30, 14, 23, 9, 32, 35, 0, 41], [30, 14, 23, 9, 32, 37, 48, 10, 43, 14], [23, 3, 34, 37, 26, 10, 33, 14, 23, 9], [39, 31, 12, 5, 43, 14, 23, 15, 16, 31], [25, 10, 43, 14, 23, 3, 18, 31, 25, 5], [33, 14, 23, 9, 39, 31, 4, 11, 33, 14], [23, 15, 29, 35, 6, 45, 30, 14, 23, 9], [29, 35, 25, 47, 43, 14, 23, 15, 32, 31], [42, 5, 33, 14, 23, 9, 16, 31, 24, 11], [43, 14, 23, 3, 17, 35, 26, 11, 43, 14], [23, 15, 16, 37, 19, 46, 33, 14, 23, 15], [32, 31, 25, 11, 43, 14, 23, 3, 39, 37], [42, 40, 33, 14, 23, 9, 32, 31, 22, 41], [43, 14, 23, 3, 16, 31, 7, 45, 30, 14], [23, 15, 18, 37, 0, 40, 33, 14, 23, 9], [16, 35, 44, 47, 33, 14, 23, 15, 34, 37], [22, 10, 43, 14, 23, 9, 29, 37, 8, 41], [30, 14, 23, 15, 34, 31, 20, 47, 30, 14], [23, 3, 28, 31, 26, 40, 33, 14, 23, 9], [34, 35, 48, 47, 43, 14, 23, 9, 17, 37], [0, 40, 33, 14, 23, 15, 39, 37, 8, 45], [30, 14, 23, 15, 28, 35, 21, 41, 30, 14], [23, 15, 17, 35, 49, 40, 33, 14, 23, 9], [29, 31, 7, 5, 30, 14, 23, 3, 28, 37], [21, 5, 30, 14, 23, 9, 29, 37, 12, 5], [43, 14, 23, 3, 1, 35, 49, 40, 33, 14], [23, 3, 2, 35, 20, 45, 33, 14, 23, 3], [18, 37, 19, 47, 43, 14, 23, 9, 1, 37], [21, 45, 30, 14, 23, 15, 38, 35, 12, 46], [33, 14, 23, 15, 18, 31, 20, 40, 43, 14], [23, 15, 18, 35, 20, 10, 33, 14, 23, 9], [28, 37, 36, 10, 30, 14, 23, 9, 28, 35], [49, 11, 30, 14, 23, 15, 2, 35, 8, 10], [33, 14, 23, 9, 32, 35, 24, 45, 33, 14], [23, 3, 38, 35, 21, 11, 30, 14, 23, 9], [1, 37, 49, 41, 30, 14, 23, 15, 38, 37], [26, 5, 43, 14, 23, 15, 39, 31, 42, 45], [30, 14, 23, 15, 17, 35, 42, 10, 30, 14], [23, 15, 38, 37, 25, 45, 30, 14, 23, 15], [34, 35, 49, 45, 30, 14, 23, 15, 34, 31], [49, 5, 30, 14, 23, 9, 34, 31, 19, 41], [33, 14, 23, 3, 1, 35, 27, 47, 33, 14], [23, 9, 28, 31, 36, 41, 30, 14, 23, 9], [34, 35, 27, 45, 30, 14, 23, 9, 16, 31], [8, 46, 30, 14, 23, 9, 16, 37, 22, 46], [43, 14, 23, 3, 28, 31, 36, 10, 30, 14], [23, 15, 32, 35, 27, 46, 33, 14, 23, 3], [39, 37, 22, 10, 30, 14, 23, 15, 2, 35], [4, 47, 33, 14, 23, 9, 2, 35, 22, 41], [43, 14, 23, 9, 29, 37, 48, 10, 30, 14], [23, 3, 18, 35, 12, 41, 43, 14, 23, 9], [38, 31, 19, 41, 30, 14, 23, 9, 1, 35], [4, 40, 33, 14, 23, 3, 1, 31, 36, 47], [33, 14, 23, 9, 16, 35, 25, 40, 30, 14], [23, 9, 1, 31, 25, 40, 30, 14, 23, 9], [28, 37, 6, 47, 43, 14, 23, 15, 28, 31], [48, 47, 33, 14, 23, 15, 16, 37, 27, 11], [30, 14, 23, 3, 16, 37, 26, 41, 30, 14], [23, 3, 16, 35, 22, 47, 30, 14, 23, 15], [1, 31, 0, 11, 30, 14, 23, 3, 34, 35], [22, 5, 33, 14, 23, 15, 28, 31, 48, 46], [33, 14, 23, 9, 1, 31, 27, 10, 43, 14], [23, 9, 16, 37, 26, 46, 43, 14, 23, 15], [28, 35, 26, 5, 30, 14, 23, 15, 34, 37], [27, 46, 43, 14, 23, 9, 28, 31, 25, 46], [33, 14, 23, 9, 17, 31, 48, 46, 43, 14], [23, 3, 28, 35, 22, 10, 33, 14, 23, 9], [16, 31, 0, 47, 33, 14, 23, 9, 34, 35], [13, 11, 33, 14, 23, 3, 39, 35, 8, 10], [43, 14, 23, 3, 38, 35, 24, 11, 43, 14], [23, 9, 39, 31, 21, 46, 30, 14, 23, 15], [2, 35, 12, 41, 30, 14, 23, 9, 28, 35], [22, 10, 30, 14, 23, 15, 2, 37, 26, 40], [33, 14, 23, 15, 17, 31, 44, 47, 43, 14], [23, 3, 39, 35, 0, 41, 30, 14, 23, 9], [38, 31, 20, 10, 43, 14, 23, 15, 29, 31], [0, 11, 43, 14, 23, 3, 1, 37, 4, 47], [43, 14, 23, 9, 18, 31, 21, 10, 43, 14], [23, 15, 34, 31, 24, 41, 33, 14, 23, 3], [28, 37, 21, 47, 30, 14, 23, 9, 38, 37], [20, 11, 43, 14, 23, 9, 32, 35, 42, 40], [33, 14, 23, 15, 16, 31, 7, 40, 30, 14], [23, 3, 29, 35, 13, 40, 33, 14, 23, 3], [32, 37, 22, 11, 43, 14, 23, 3, 1, 35], [8, 11, 43, 14, 23, 9, 38, 35, 25, 40], [33, 14, 23, 15, 32, 37, 27, 10, 30, 14], [23, 9, 38, 37, 27, 11, 30, 14, 23, 9], [38, 31, 6, 45, 33, 14, 23, 3, 2, 37], [44, 47, 30, 14, 23, 15, 2, 37, 19, 47], [43, 14, 23, 9, 1, 37, 27, 40, 30, 14], [23, 9, 38, 31, 19, 40, 43, 14, 23, 3], [1, 31, 36, 11, 43, 14, 23, 9, 39, 31], [26, 10, 43, 14, 23, 9, 16, 31, 19, 40], [30, 14, 23, 15, 38, 31, 6, 47, 30, 14], [23, 15, 18, 35, 21, 46, 33, 14, 23, 15], [2, 37, 25, 47, 43, 14, 23, 3, 28, 37], [26, 5, 43, 14, 23, 9, 28, 31, 42, 10], [43, 14, 23, 15, 29, 35, 19, 40, 43, 14], [23, 15, 28, 35, 48, 40, 33, 14, 23, 15], [29, 31, 7, 11, 30, 14, 23, 3, 34, 35], [0, 40, 33, 14, 23, 9, 17, 37, 6, 10], [30, 14, 23, 9, 29, 35, 26, 41, 43, 14], [23, 15, 1, 37, 44, 40, 30, 14, 23, 9], [17, 35, 13, 47, 30, 14, 23, 3, 16, 37], [22, 41, 33, 14, 23, 9, 2, 31, 36, 41], [43, 14, 23, 9, 17, 37, 7, 10, 33, 14], [23, 3, 39, 31, 26, 11, 30, 14, 23, 3], [39, 37, 13, 47, 30, 14, 23, 15, 39, 37], [20, 10, 30, 14, 23, 3, 2, 37, 8, 45], [30, 14, 23, 15, 32, 31, 4, 10, 43, 14], [23, 9, 39, 35, 19, 10, 30, 14, 23, 15], [16, 35, 27, 46, 43, 14, 23, 15, 17, 35], [0, 46, 43, 14, 23, 15, 1, 35, 4, 41], [33, 14, 23, 15, 32, 35, 13, 11, 33, 14], [23, 9, 17, 37, 20, 11, 33, 14, 23, 15], [39, 37, 4, 40, 33, 14, 23, 3, 2, 37], [24, 46, 43, 14, 23, 3, 16, 37, 44, 5], [43, 14, 23, 3, 28, 31, 13, 45, 30, 14], [23, 3, 18, 35, 26, 46, 33, 14, 23, 9], [17, 35, 24, 41, 33, 14, 23, 15, 1, 31], [13, 11, 30, 14, 23, 9, 39, 31, 44, 47], [33, 14, 23, 3, 39, 35, 8, 45, 30, 14], [23, 3, 39, 37, 42, 10, 43, 14, 23, 9], [39, 35, 27, 45, 33, 14, 23, 9, 28, 35], [48, 11, 43, 14, 23, 9, 39, 31, 36, 41], [43, 14, 23, 15, 1, 35, 26, 45, 30, 14], [23, 15, 29, 35, 25, 41, 30, 14, 23, 3], [28, 31, 27, 45, 33, 14, 23, 3, 34, 31], [13, 11, 43, 14, 23, 3, 29, 37, 13, 10], [43, 14, 23, 3, 34, 37, 44, 11, 43, 14], [23, 15, 39, 35, 7, 10, 30, 14, 23, 3], [18, 37, 26, 11, 33, 14, 23, 3, 17, 35], [26, 47, 30, 14, 23, 3, 18, 35, 6, 47], [33, 14, 23, 3, 18, 31, 6, 47, 43, 14], [23, 3, 16, 31, 7, 46, 43, 14, 23, 9], [1, 35, 8, 47, 33, 14, 23, 15, 18, 37], [22, 5, 30, 14, 23, 9, 1, 37, 48, 10], [30, 14, 23, 15, 34, 37, 22, 10, 33, 14], [23, 15, 1, 31, 21, 47, 43, 14, 23, 9], [16, 31, 48, 47, 30, 14, 23, 9, 17, 37], [24, 11, 43, 14, 23, 3, 1, 35, 48, 45], [30, 14, 23, 9, 32, 37, 8, 40, 33, 14], [23, 9, 16, 31, 8, 47, 30, 14, 23, 15], [18, 35, 22, 10, 30, 14, 23, 9, 2, 37]]
        if how_many_parts == 5:
            picked_model = [[13, 46, 30], [14, 23, 9], [38, 35, 36], [46, 43, 14], [23, 15, 2], [37, 25, 40], [43, 14, 23], [9, 28, 37], [48, 41, 33], [14, 23, 9], [18, 31, 42], [10, 33, 14], [23, 15, 39], [37, 21, 10], [30, 14, 23], [3, 2, 37], [22, 40, 33], [14, 23, 9], [2, 31, 8], [46, 43, 14], [23, 9, 16], [37, 27, 47], [30, 14, 23], [15, 2, 35], [22, 45, 43], [14, 23, 9], [29, 37, 6], [11, 43, 14], [23, 9, 1], [35, 48, 47], [30, 14, 23], [15, 2, 35], [4, 40, 33], [14, 23, 9], [18, 31, 44], [46, 30, 14], [23, 3, 18], [37, 6, 46], [30, 14, 23], [3, 28, 37], [22, 47, 43], [14, 23, 15], [34, 35, 20], [41, 30, 14], [23, 3, 38], [35, 4, 45], [43, 14, 23], [9, 18, 31], [21, 47, 33], [14, 23, 15], [16, 37, 20], [46, 33, 14], [23, 3, 39], [37, 25, 40], [33, 14, 23], [9, 17, 31], [0, 47, 43], [14, 23, 3], [29, 35, 42], [47, 43, 14], [23, 15, 28], [37, 4, 46], [33, 14, 23], [9, 39, 31], [12, 5, 43], [14, 23, 15], [18, 35, 49], [45, 30, 14], [23, 15, 39], [37, 44, 5], [33, 14, 23], [3, 16, 31], [6, 10, 43], [14, 23, 9], [32, 35, 4], [10, 30, 14], [23, 3, 16], [37, 22, 11], [30, 14, 23], [9, 1, 37], [44, 45, 30], [14, 23, 15], [2, 37, 49], [41, 43, 14], [23, 9, 17], [31, 25, 10], [30, 14, 23], [15, 28, 35], [27, 11, 43], [14, 23, 15], [1, 31, 36], [5, 30, 14], [23, 15, 18], [37, 48, 45], [30, 14, 23], [9, 28, 31], [24, 40, 30], [14, 23, 15], [39, 37, 13], [11, 33, 14], [23, 9, 39], [35, 8, 10], [43, 14, 23], [3, 34, 31], [49, 5, 43], [14, 23, 9], [32, 35, 21], [40, 33, 14], [23, 9, 34], [37, 36, 47], [30, 14, 23], [3, 38, 31], [21, 46, 30], [14, 23, 3], [28, 31, 7], [5, 43, 14], [23, 9, 1], [37, 22, 41], [33, 14, 23], [15, 17, 35], [8, 46, 30], [14, 23, 3], [39, 37, 27], [45, 33, 14], [23, 9, 18], [37, 13, 41], [33, 14, 23], [9, 18, 35], [13, 47, 43], [14, 23, 15], [2, 31, 24], [40, 43, 14], [23, 3, 34], [31, 48, 5], [43, 14, 23], [3, 29, 31], [26, 40, 43], [14, 23, 9], [2, 31, 13], [47, 43, 14], [23, 15, 39], [31, 27, 5], [33, 14, 23], [15, 38, 31], [36, 10, 33], [14, 23, 15], [32, 35, 13], [45, 30, 14], [23, 9, 1], [31, 13, 41], [33, 14, 23], [3, 17, 37], [24, 40, 33], [14, 23, 15], [18, 31, 13], [40, 33, 14], [23, 15, 28], [37, 42, 47], [30, 14, 23], [9, 32, 31], [8, 45, 33], [14, 23, 3], [2, 35, 36], [46, 33, 14], [23, 3, 29], [37, 26, 11], [30, 14, 23], [9, 38, 35], [24, 41, 30], [14, 23, 15], [34, 31, 8], [46, 33, 14], [23, 15, 28], [31, 20, 10], [30, 14, 23], [3, 28, 37], [4, 41, 30], [14, 23, 15], [29, 31, 24], [40, 33, 14], [23, 15, 38], [35, 36, 46], [30, 14, 23], [15, 18, 37], [49, 45, 43], [14, 23, 9], [2, 31, 27], [41, 43, 14], [23, 9, 34], [35, 4, 45], [30, 14, 23], [9, 34, 35], [13, 11, 30], [14, 23, 3], [38, 31, 12], [10, 33, 14], [23, 15, 34], [31, 25, 40], [33, 14, 23], [15, 18, 37], [4, 47, 33], [14, 23, 15], [18, 31, 49], [41, 30, 14], [23, 15, 16], [37, 4, 41], [30, 14, 23], [15, 29, 31], [49, 40, 43], [14, 23, 9], [16, 35, 4], [47, 30, 14], [23, 3, 39], [37, 36, 11], [33, 14, 23], [3, 39, 37], [8, 45, 43], [14, 23, 9], [34, 37, 13], [11, 33, 14], [23, 9, 34], [31, 4, 41], [30, 14, 23], [15, 28, 37], [12, 46, 30], [14, 23, 15], [18, 35, 7], [5, 30, 14], [23, 15, 29], [31, 21, 10], [30, 14, 23], [15, 16, 35], [44, 45, 43], [14, 23, 15], [32, 37, 6], [40, 33, 14], [23, 9, 18], [35, 13, 11], [43, 14, 23], [9, 2, 37], [4, 10, 33], [14, 23, 3], [29, 35, 13], [45, 33, 14], [23, 3, 32], [35, 24, 11], [43, 14, 23], [15, 2, 35], [44, 47, 43], [14, 23, 3], [34, 37, 26], [11, 43, 14], [23, 15, 39], [31, 0, 5], [30, 14, 23], [15, 32, 35], [49, 46, 43], [14, 23, 3], [18, 35, 44], [46, 43, 14], [23, 9, 17], [37, 8, 45], [33, 14, 23], [9, 38, 35], [48, 46, 33], [14, 23, 9], [16, 31, 27], [5, 30, 14], [23, 15, 16], [37, 12, 40], [33, 14, 23], [3, 1, 37], [49, 47, 33], [14, 23, 9], [17, 35, 13], [10, 33, 14], [23, 3, 32], [35, 7, 41], [33, 14, 23], [3, 38, 35], [19, 45, 33], [14, 23, 15], [18, 31, 48], [5, 43, 14], [23, 15, 39], [35, 0, 41], [30, 14, 23], [3, 38, 35], [48, 41, 43], [14, 23, 15], [38, 31, 21], [10, 33, 14], [23, 3, 16], [35, 13, 47], [43, 14, 23], [9, 2, 35], [26, 41, 43], [14, 23, 3], [34, 31, 6], [41, 33, 14], [23, 3, 2], [31, 42, 47], [30, 14, 23], [15, 16, 31], [27, 45, 33], [14, 23, 3], [1, 31, 44], [5, 33, 14], [23, 15, 38], [35, 0, 11], [43, 14, 23], [9, 32, 35], [27, 41, 33], [14, 23, 3], [1, 35, 27], [45, 33, 14], [23, 9, 38], [31, 13, 5], [43, 14, 23], [3, 28, 37], [48, 47, 30], [14, 23, 3], [34, 31, 6], [41, 43, 14], [23, 3, 38], [31, 21, 5], [30, 14, 23], [15, 16, 35], [27, 45, 43], [14, 23, 15], [28, 37, 22], [45, 43, 14], [23, 3, 2], [35, 20, 10], [33, 14, 23], [15, 18, 35], [49, 46, 33], [14, 23, 3], [39, 37, 0], [5, 43, 14], [23, 3, 38], [35, 19, 45], [30, 14, 23], [3, 29, 31], [25, 45, 33], [14, 23, 3], [34, 35, 22], [10, 30, 14], [23, 15, 17], [31, 19, 45], [43, 14, 23], [9, 18, 35], [6, 45, 43], [14, 23, 3], [2, 35, 27], [5, 30, 14], [23, 3, 39], [37, 12, 47], [33, 14, 23], [15, 29, 37], [13, 45, 30], [14, 23, 15], [34, 35, 20], [40, 43, 14], [23, 9, 32], [31, 4, 46], [43, 14, 23], [3, 1, 35], [25, 10, 33], [14, 23, 15], [32, 37, 24], [41, 30, 14], [23, 9, 2], [37, 19, 5], [30, 14, 23], [15, 38, 37], [7, 10, 43], [14, 23, 15], [39, 31, 4], [5, 43, 14], [23, 15, 16], [37, 27, 5], [30, 14, 23], [3, 18, 35], [49, 45, 33], [14, 23, 9], [2, 31, 25], [5, 43, 14], [23, 3, 39], [37, 44, 40], [43, 14, 23], [3, 38, 31], [25, 10, 43], [14, 23, 15], [32, 31, 26], [5, 43, 14], [23, 9, 2], [35, 20, 45], [33, 14, 23], [3, 2, 31]]

    if alpha == 0.6:
        if how_many_parts == 15:
            picked_model = [[35, 31, 21, 12, 38, 11, 0, 6], [24, 42, 23, 12, 41, 37, 3, 6], [35, 10, 22, 12, 41, 48, 0, 6], [13, 4, 23, 15, 38, 11, 3, 6], [35, 28, 27, 12, 26, 44, 2, 6], [25, 9, 39, 15, 38, 19, 5, 8], [35, 18, 21, 12, 38, 49, 7, 6], [32, 42, 30, 15, 20, 45, 3, 6], [13, 10, 39, 15, 26, 1, 3, 6], [25, 31, 40, 15, 26, 34, 5, 8], [25, 18, 39, 12, 38, 44, 2, 8], [25, 42, 30, 12, 17, 43, 5, 6], [16, 42, 21, 15, 26, 43, 3, 8], [16, 31, 27, 15, 17, 46, 5, 8], [16, 9, 40, 12, 17, 34, 5, 6], [25, 10, 40, 15, 41, 1, 3, 6], [16, 14, 27, 12, 26, 47, 5, 6], [13, 28, 30, 12, 17, 46, 2, 8], [32, 29, 22, 15, 20, 44, 3, 6], [35, 28, 40, 12, 17, 46, 5, 6], [35, 10, 21, 12, 38, 46, 3, 8], [13, 18, 23, 12, 41, 44, 3, 8], [16, 36, 22, 12, 41, 19, 0, 8], [16, 36, 22, 15, 20, 49, 2, 6], [16, 18, 21, 15, 41, 46, 3, 6], [35, 14, 30, 12, 17, 43, 2, 8], [32, 28, 39, 12, 20, 34, 2, 8], [24, 29, 39, 15, 20, 1, 3, 6], [35, 36, 40, 15, 20, 34, 5, 6], [32, 28, 23, 15, 41, 45, 5, 6], [25, 14, 40, 15, 38, 47, 2, 8], [25, 42, 27, 15, 38, 45, 5, 8], [25, 33, 21, 12, 20, 44, 3, 6], [35, 28, 30, 12, 17, 47, 7, 8], [35, 28, 30, 12, 38, 37, 0, 8], [32, 31, 23, 15, 17, 19, 7, 8], [35, 29, 30, 15, 26, 45, 5, 6], [16, 9, 30, 15, 41, 49, 0, 8], [13, 18, 23, 12, 38, 49, 2, 6], [25, 10, 21, 12, 41, 1, 7, 6], [13, 14, 23, 12, 41, 45, 5, 6], [13, 31, 23, 12, 17, 47, 5, 8], [32, 14, 40, 15, 20, 11, 3, 8], [24, 31, 21, 15, 26, 44, 0, 8], [35, 18, 23, 15, 26, 19, 2, 8], [35, 31, 39, 12, 38, 34, 5, 6], [24, 31, 40, 15, 38, 49, 2, 6], [32, 18, 22, 12, 20, 46, 5, 6], [13, 29, 22, 15, 41, 49, 2, 6], [32, 31, 30, 15, 20, 48, 7, 8], [35, 31, 21, 15, 17, 1, 7, 8], [25, 29, 21, 12, 26, 45, 5, 6], [32, 10, 39, 12, 41, 34, 3, 6], [25, 29, 40, 12, 26, 43, 3, 6], [35, 42, 23, 15, 17, 47, 7, 6], [16, 18, 40, 15, 20, 45, 0, 8], [35, 18, 22, 12, 26, 46, 2, 6], [13, 31, 39, 15, 20, 43, 7, 6], [24, 36, 21, 15, 38, 49, 3, 6], [16, 36, 22, 12, 17, 49, 7, 8], [35, 33, 23, 12, 20, 49, 0, 6], [16, 10, 23, 12, 17, 19, 5, 6], [25, 33, 23, 12, 20, 45, 7, 6], [25, 42, 21, 12, 26, 48, 7, 6], [35, 29, 23, 15, 17, 44, 2, 6], [32, 9, 23, 15, 38, 45, 3, 8], [24, 36, 39, 15, 38, 11, 7, 6], [16, 31, 30, 12, 26, 49, 2, 8], [13, 29, 22, 12, 38, 45, 0, 6], [35, 42, 22, 15, 26, 45, 5, 8], [24, 42, 30, 15, 38, 47, 2, 6], [13, 10, 22, 12, 17, 48, 0, 8], [24, 18, 40, 15, 38, 43, 0, 8], [16, 4, 39, 12, 38, 49, 7, 6], [35, 4, 30, 12, 41, 48, 5, 8], [16, 18, 30, 15, 41, 46, 3, 8], [24, 14, 27, 15, 20, 34, 5, 6], [16, 10, 27, 12, 26, 48, 5, 6], [16, 4, 40, 15, 41, 37, 2, 6], [25, 14, 39, 15, 38, 34, 7, 8], [24, 9, 27, 12, 20, 48, 5, 8], [13, 33, 40, 15, 41, 49, 5, 8], [25, 31, 30, 15, 20, 34, 0, 8], [16, 9, 39, 15, 26, 44, 2, 8], [16, 9, 27, 12, 38, 43, 0, 6], [25, 29, 40, 15, 41, 46, 3, 8], [25, 42, 23, 15, 38, 49, 7, 6], [16, 29, 40, 15, 17, 43, 3, 6], [24, 4, 21, 15, 20, 47, 5, 6], [16, 18, 40, 15, 41, 48, 0, 8], [24, 18, 40, 12, 41, 43, 7, 6], [35, 18, 23, 12, 38, 43, 5, 6], [35, 10, 27, 15, 17, 47, 7, 8], [35, 9, 21, 15, 41, 49, 0, 8], [35, 14, 22, 12, 26, 46, 5, 6], [25, 42, 22, 15, 20, 46, 5, 6], [13, 31, 23, 12, 26, 44, 2, 6], [35, 28, 40, 12, 41, 47, 3, 6], [35, 4, 23, 15, 20, 44, 7, 8], [35, 31, 27, 12, 17, 45, 2, 8], [16, 10, 23, 12, 20, 48, 5, 6], [13, 9, 27, 12, 26, 1, 3, 8], [32, 28, 21, 12, 38, 48, 2, 6], [24, 18, 22, 12, 41, 34, 3, 8], [32, 31, 21, 15, 17, 49, 3, 6], [32, 28, 39, 12, 41, 45, 7, 6], [32, 42, 39, 12, 26, 45, 5, 6], [13, 14, 23, 12, 20, 19, 5, 6], [16, 31, 39, 12, 26, 19, 0, 8], [35, 33, 21, 15, 20, 34, 2, 6], [13, 4, 40, 15, 17, 44, 7, 8], [25, 10, 27, 15, 38, 46, 2, 6], [16, 28, 27, 15, 26, 44, 5, 8], [32, 28, 39, 15, 20, 1, 5, 6], [24, 4, 22, 12, 38, 47, 2, 8], [16, 14, 21, 15, 26, 48, 2, 6], [25, 29, 27, 12, 38, 11, 7, 6], [24, 4, 40, 15, 41, 19, 5, 8], [13, 18, 27, 15, 26, 47, 5, 8], [16, 31, 23, 15, 26, 44, 7, 8], [35, 29, 30, 15, 41, 44, 2, 6], [35, 10, 22, 12, 41, 48, 5, 6], [13, 31, 30, 15, 20, 1, 2, 6], [25, 18, 30, 12, 17, 11, 0, 8], [25, 18, 39, 12, 20, 47, 0, 8], [35, 36, 27, 12, 17, 11, 5, 6], [16, 33, 22, 15, 20, 37, 3, 6], [32, 33, 22, 12, 20, 43, 7, 6], [35, 28, 39, 15, 26, 45, 2, 8], [32, 33, 21, 15, 38, 1, 0, 8], [16, 31, 39, 15, 41, 45, 7, 6], [25, 14, 27, 12, 17, 11, 7, 6], [25, 4, 23, 15, 41, 43, 7, 6], [35, 4, 21, 15, 17, 43, 0, 8], [24, 10, 27, 12, 26, 43, 0, 6], [24, 14, 22, 15, 26, 45, 2, 6], [24, 10, 21, 15, 26, 11, 3, 8], [13, 14, 30, 12, 17, 46, 7, 6], [13, 10, 39, 12, 38, 48, 5, 6], [16, 14, 21, 12, 26, 43, 2, 8], [13, 33, 21, 15, 38, 34, 7, 8], [24, 9, 27, 15, 26, 44, 3, 6], [32, 28, 21, 15, 41, 47, 3, 6], [24, 29, 40, 12, 26, 45, 5, 8], [32, 29, 40, 15, 17, 44, 0, 6], [24, 10, 27, 15, 41, 47, 0, 8], [24, 18, 27, 12, 41, 43, 3, 6], [35, 10, 21, 15, 38, 1, 5, 8], [32, 42, 30, 15, 17, 11, 2, 8], [13, 4, 21, 12, 26, 34, 2, 6], [24, 31, 23, 15, 20, 45, 5, 6], [24, 18, 22, 15, 20, 37, 2, 8], [35, 9, 21, 15, 17, 47, 3, 6], [35, 29, 30, 15, 38, 43, 0, 6], [32, 33, 23, 12, 41, 34, 0, 6], [32, 29, 27, 12, 20, 11, 0, 8], [32, 28, 27, 15, 20, 49, 5, 8], [16, 18, 40, 15, 17, 46, 7, 6], [35, 31, 30, 15, 38, 49, 0, 6], [16, 36, 23, 15, 38, 49, 5, 6], [13, 36, 22, 15, 26, 11, 5, 6], [35, 14, 22, 12, 38, 46, 2, 8], [24, 31, 27, 12, 41, 34, 3, 6], [16, 29, 23, 12, 17, 19, 7, 6], [13, 10, 22, 12, 20, 37, 2, 8], [16, 28, 22, 12, 20, 1, 5, 6], [25, 36, 30, 12, 20, 45, 0, 8], [35, 18, 40, 12, 41, 11, 7, 6], [32, 14, 27, 12, 26, 43, 0, 6], [35, 42, 27, 15, 17, 45, 2, 8], [13, 31, 23, 12, 20, 43, 2, 6], [16, 36, 27, 15, 41, 46, 0, 8], [32, 28, 27, 15, 41, 37, 3, 8], [16, 28, 23, 15, 26, 19, 2, 8], [25, 29, 27, 15, 26, 1, 3, 8], [13, 33, 30, 15, 26, 46, 0, 8], [32, 18, 23, 15, 41, 34, 5, 6], [13, 28, 22, 12, 41, 44, 5, 6], [32, 29, 21, 12, 41, 47, 3, 8], [16, 4, 21, 15, 41, 48, 7, 6], [25, 29, 22, 12, 41, 11, 0, 6], [24, 42, 30, 12, 20, 1, 0, 6], [35, 29, 21, 15, 38, 37, 2, 8], [13, 10, 40, 12, 20, 11, 5, 6], [24, 31, 22, 15, 20, 34, 0, 8], [32, 10, 27, 15, 38, 48, 0, 8], [32, 14, 27, 12, 26, 1, 5, 6], [32, 28, 22, 15, 17, 37, 0, 8], [35, 31, 23, 15, 26, 34, 7, 6], [32, 10, 21, 15, 41, 45, 5, 8], [13, 42, 39, 12, 38, 49, 3, 6], [35, 28, 40, 12, 17, 49, 3, 6], [25, 28, 21, 15, 26, 48, 0, 6], [16, 9, 30, 15, 20, 45, 5, 8], [13, 33, 40, 15, 26, 44, 5, 8], [25, 14, 23, 12, 26, 34, 7, 8], [16, 4, 21, 15, 26, 46, 5, 8], [25, 33, 21, 12, 17, 44, 3, 8], [25, 18, 21, 15, 20, 48, 5, 6], [32, 4, 21, 12, 26, 34, 3, 8], [24, 29, 39, 15, 17, 43, 5, 8], [35, 36, 30, 12, 38, 1, 3, 6], [13, 42, 27, 12, 38, 37, 3, 8], [16, 18, 27, 15, 41, 44, 7, 6], [16, 36, 39, 12, 17, 44, 0, 8], [13, 14, 40, 15, 38, 34, 3, 8], [16, 9, 30, 12, 20, 47, 0, 8], [24, 29, 21, 15, 41, 1, 2, 6], [32, 28, 40, 15, 38, 46, 3, 8], [24, 31, 27, 15, 41, 37, 0, 6], [35, 28, 27, 12, 20, 37, 3, 6], [32, 42, 22, 12, 17, 47, 0, 6], [13, 42, 39, 12, 20, 45, 5, 6], [24, 9, 27, 12, 26, 49, 0, 8], [35, 36, 40, 12, 20, 37, 7, 6], [25, 28, 39, 12, 38, 44, 0, 8], [16, 10, 27, 12, 20, 11, 7, 8], [16, 4, 40, 15, 17, 47, 7, 8], [16, 10, 23, 12, 17, 37, 3, 8], [25, 10, 40, 12, 41, 37, 5, 8], [16, 14, 22, 15, 20, 44, 3, 8], [16, 42, 21, 15, 26, 44, 5, 6], [25, 18, 22, 12, 20, 1, 5, 6], [25, 4, 23, 12, 38, 48, 0, 8], [35, 28, 30, 15, 38, 37, 3, 8], [24, 42, 21, 12, 17, 49, 7, 8], [16, 18, 39, 15, 26, 43, 3, 6], [32, 42, 23, 12, 20, 43, 5, 8], [35, 36, 27, 15, 41, 34, 5, 6], [35, 28, 21, 12, 26, 34, 5, 6], [35, 14, 22, 12, 17, 11, 7, 6], [25, 10, 40, 12, 26, 45, 2, 6], [32, 9, 30, 15, 26, 47, 7, 6], [13, 29, 22, 12, 17, 46, 5, 8], [25, 29, 27, 15, 17, 46, 2, 6], [35, 18, 40, 15, 26, 46, 5, 6], [16, 28, 30, 15, 26, 43, 7, 6], [16, 29, 39, 15, 26, 37, 7, 8], [13, 29, 27, 12, 38, 45, 5, 6], [16, 42, 30, 12, 38, 47, 3, 6], [16, 29, 21, 15, 38, 19, 3, 6], [24, 4, 21, 12, 41, 37, 7, 6], [35, 14, 40, 15, 41, 49, 0, 8], [16, 29, 39, 12, 20, 43, 2, 6], [13, 31, 23, 12, 17, 43, 0, 8], [24, 42, 39, 15, 38, 37, 2, 6], [24, 10, 23, 15, 41, 34, 0, 6], [24, 36, 30, 12, 38, 34, 2, 8], [35, 42, 39, 12, 38, 37, 3, 6], [13, 18, 23, 15, 26, 37, 3, 6], [13, 29, 27, 15, 20, 43, 5, 6], [32, 29, 39, 15, 20, 11, 2, 6], [25, 4, 40, 15, 17, 48, 0, 8], [24, 14, 23, 15, 38, 43, 3, 8], [35, 36, 21, 12, 41, 43, 0, 6], [35, 31, 21, 15, 38, 19, 7, 6], [24, 33, 40, 12, 20, 48, 2, 6], [35, 10, 23, 12, 41, 47, 3, 6], [24, 33, 27, 12, 20, 45, 2, 8], [24, 9, 22, 12, 20, 47, 5, 8], [16, 4, 30, 15, 17, 48, 2, 6], [25, 31, 23, 12, 17, 43, 7, 8], [32, 4, 21, 15, 17, 34, 0, 6], [16, 9, 27, 12, 20, 47, 2, 6], [16, 28, 22, 12, 17, 1, 5, 6], [16, 4, 30, 15, 20, 49, 3, 6], [16, 18, 21, 15, 26, 44, 2, 6], [24, 28, 23, 15, 38, 47, 5, 6], [13, 18, 22, 15, 20, 49, 2, 8], [32, 33, 27, 15, 41, 11, 3, 8], [13, 33, 21, 15, 38, 43, 0, 8], [13, 33, 39, 15, 41, 37, 3, 6], [24, 31, 27, 12, 20, 49, 7, 6], [24, 4, 39, 15, 38, 44, 7, 6], [32, 4, 23, 12, 17, 49, 5, 8], [25, 14, 40, 12, 41, 47, 0, 8], [13, 33, 23, 15, 26, 34, 2, 8], [16, 9, 40, 12, 17, 49, 0, 6], [32, 9, 27, 12, 17, 37, 3, 8], [13, 9, 40, 15, 41, 1, 2, 8], [13, 18, 30, 15, 41, 43, 7, 8], [25, 31, 40, 15, 17, 44, 5, 8], [25, 42, 23, 12, 41, 44, 0, 8], [32, 36, 30, 15, 26, 44, 2, 8], [25, 29, 27, 15, 20, 11, 3, 6], [35, 14, 21, 12, 20, 47, 7, 8], [13, 28, 30, 12, 41, 48, 7, 6], [16, 36, 40, 15, 38, 43, 3, 8], [25, 4, 22, 15, 38, 46, 3, 6], [24, 9, 21, 15, 41, 45, 5, 8], [32, 9, 40, 12, 20, 11, 2, 6], [13, 42, 27, 15, 38, 11, 5, 6], [24, 10, 23, 15, 26, 34, 0, 8], [24, 31, 30, 15, 38, 19, 5, 6], [35, 29, 21, 12, 26, 44, 5, 6], [24, 33, 40, 15, 26, 43, 5, 6], [16, 36, 21, 12, 38, 43, 2, 6], [32, 9, 23, 15, 26, 48, 3, 6], [13, 4, 27, 12, 26, 48, 2, 8], [13, 31, 40, 12, 38, 19, 3, 6], [32, 42, 21, 15, 38, 48, 0, 8], [13, 31, 27, 15, 41, 19, 2, 6], [32, 9, 30, 15, 41, 43, 0, 6], [24, 36, 40, 12, 26, 46, 5, 6], [24, 31, 22, 12, 17, 19, 5, 8], [35, 31, 40, 12, 20, 11, 2, 8], [25, 28, 39, 15, 38, 43, 5, 6], [35, 31, 30, 12, 26, 49, 5, 8], [25, 29, 39, 15, 26, 37, 5, 8], [32, 42, 27, 12, 41, 43, 5, 6], [32, 33, 27, 12, 20, 19, 7, 6], [16, 29, 40, 15, 20, 45, 3, 6], [35, 36, 23, 12, 20, 47, 3, 8], [32, 36, 30, 15, 20, 48, 5, 8], [24, 36, 40, 15, 17, 37, 7, 6], [13, 9, 39, 15, 41, 48, 0, 8], [35, 4, 21, 15, 41, 34, 3, 8], [35, 31, 22, 12, 41, 44, 5, 8], [32, 9, 22, 15, 20, 44, 7, 6], [13, 4, 30, 15, 17, 1, 5, 6], [25, 14, 39, 12, 38, 48, 7, 6], [16, 28, 23, 12, 38, 45, 0, 6], [25, 42, 21, 12, 26, 45, 2, 6], [16, 4, 39, 15, 26, 49, 5, 6], [24, 9, 23, 15, 41, 19, 3, 6], [13, 31, 21, 15, 38, 19, 3, 6], [24, 28, 23, 12, 20, 47, 0, 6], [35, 42, 40, 15, 26, 47, 0, 6], [24, 4, 21, 15, 26, 1, 5, 6], [35, 14, 27, 12, 20, 46, 0, 8], [32, 42, 22, 12, 26, 48, 3, 6], [16, 29, 30, 15, 17, 47, 3, 6], [13, 42, 27, 15, 20, 37, 0, 8], [16, 36, 39, 15, 20, 47, 3, 8], [13, 4, 30, 15, 17, 34, 0, 6], [13, 14, 40, 12, 38, 1, 5, 8], [35, 36, 30, 15, 17, 19, 5, 6], [13, 4, 30, 12, 17, 19, 2, 8], [24, 18, 27, 15, 26, 47, 5, 6], [24, 4, 21, 15, 41, 34, 5, 8], [35, 33, 40, 15, 38, 34, 3, 6], [16, 31, 30, 12, 17, 49, 3, 8], [25, 31, 39, 15, 26, 19, 5, 8], [25, 31, 40, 12, 20, 1, 0, 8], [13, 36, 39, 15, 38, 43, 0, 6], [35, 28, 23, 15, 26, 47, 7, 6], [16, 9, 39, 12, 38, 34, 0, 6], [16, 4, 22, 15, 20, 11, 5, 6], [35, 14, 21, 12, 20, 49, 2, 6], [32, 42, 21, 15, 17, 37, 2, 6], [16, 4, 39, 12, 20, 46, 3, 8], [32, 33, 22, 12, 41, 46, 5, 8], [24, 28, 21, 15, 20, 47, 3, 6], [16, 18, 27, 12, 20, 46, 2, 6], [16, 36, 22, 12, 38, 1, 5, 8], [24, 42, 23, 12, 20, 34, 5, 6], [35, 33, 40, 12, 38, 47, 5, 8], [13, 10, 30, 15, 26, 44, 0, 8], [35, 4, 39, 12, 38, 11, 0, 6], [13, 36, 30, 12, 38, 48, 0, 6], [16, 42, 30, 12, 41, 46, 3, 6], [32, 18, 40, 12, 17, 44, 3, 8], [35, 18, 40, 15, 38, 47, 7, 8], [25, 31, 40, 12, 26, 48, 0, 8], [32, 14, 21, 12, 26, 43, 5, 8], [32, 4, 39, 15, 20, 1, 2, 6], [35, 18, 23, 15, 38, 37, 0, 8], [24, 29, 22, 15, 26, 45, 2, 6], [24, 42, 23, 12, 17, 11, 0, 8], [25, 10, 23, 12, 41, 34, 5, 8], [25, 31, 40, 15, 26, 34, 3, 6], [35, 42, 30, 12, 41, 1, 7, 6], [16, 29, 22, 15, 20, 37, 3, 6], [16, 36, 21, 12, 38, 37, 7, 6], [32, 42, 40, 12, 17, 37, 5, 6], [16, 33, 40, 12, 20, 34, 0, 8], [13, 31, 22, 12, 17, 46, 3, 8], [35, 33, 30, 15, 20, 46, 7, 6], [32, 14, 27, 12, 17, 49, 7, 8], [32, 31, 22, 12, 26, 49, 0, 8], [13, 31, 39, 12, 41, 34, 2, 6], [35, 10, 39, 12, 38, 43, 7, 6], [16, 4, 40, 15, 38, 43, 3, 8], [13, 33, 22, 12, 26, 1, 0, 6], [13, 10, 27, 12, 26, 48, 0, 8], [32, 31, 23, 15, 17, 46, 0, 8], [25, 9, 22, 15, 38, 44, 0, 8], [35, 42, 40, 15, 20, 1, 5, 6], [32, 42, 27, 15, 17, 45, 5, 8], [25, 18, 27, 15, 38, 1, 0, 8], [24, 10, 21, 12, 38, 43, 0, 6], [32, 28, 30, 12, 38, 1, 7, 6], [35, 33, 30, 12, 17, 45, 7, 6], [32, 33, 39, 12, 38, 47, 2, 6], [32, 33, 23, 15, 20, 11, 0, 6], [35, 18, 22, 15, 38, 37, 5, 8], [35, 10, 40, 12, 38, 34, 2, 8], [32, 42, 40, 15, 26, 49, 3, 6], [13, 42, 23, 12, 38, 48, 3, 6], [35, 36, 21, 15, 17, 47, 2, 8]]
        if how_many_parts == 10:
            picked_model = [[35, 10, 30, 15, 20], [1, 2, 8, 16, 10], [23, 12, 26, 46, 3], [6, 13, 42, 40, 15], [20, 1, 3, 8, 32], [36, 30, 12, 20, 45], [7, 6, 24, 18, 40], [15, 41, 46, 7, 6], [35, 9, 21, 12, 38], [34, 5, 6, 25, 29], [22, 12, 17, 46, 0], [8, 32, 29, 22, 12], [17, 48, 2, 6, 24], [18, 30, 12, 41, 1], [3, 8, 25, 28, 27], [15, 20, 49, 3, 6], [16, 31, 40, 15, 38], [37, 5, 6, 16, 10], [30, 15, 41, 44, 2], [6, 32, 31, 39, 15], [20, 43, 3, 6, 13], [42, 30, 12, 20, 44], [7, 6, 35, 31, 23], [12, 26, 46, 3, 8], [13, 36, 23, 12, 38], [1, 5, 6, 16, 14], [30, 15, 41, 43, 5], [6, 13, 10, 39, 15], [38, 48, 7, 8, 25], [4, 30, 12, 17, 11], [5, 6, 32, 4, 40], [12, 26, 34, 5, 6], [32, 10, 39, 12, 38], [49, 0, 6, 16, 10], [30, 15, 38, 19, 5], [8, 35, 31, 39, 12], [26, 49, 5, 6, 32], [18, 21, 15, 26, 1], [3, 6, 16, 10, 40], [12, 41, 49, 7, 8], [24, 9, 30, 15, 17], [11, 2, 8, 35, 10], [30, 15, 26, 46, 3], [8, 16, 28, 40, 15], [20, 43, 7, 6, 25], [9, 39, 12, 17, 34], [5, 6, 13, 4, 27], [12, 41, 48, 3, 6], [35, 14, 39, 12, 38], [45, 5, 6, 32, 42], [40, 15, 38, 1, 5], [6, 32, 10, 22, 15], [20, 19, 0, 6, 16], [14, 21, 15, 20, 1], [2, 6, 35, 36, 40], [15, 20, 34, 7, 8], [16, 36, 30, 15, 20], [19, 7, 6, 25, 33], [30, 15, 17, 34, 5], [6, 35, 28, 40, 15], [38, 19, 2, 6, 13], [36, 23, 12, 41, 49], [2, 6, 13, 4, 30], [15, 26, 43, 0, 8], [13, 33, 22, 15, 38], [45, 3, 8, 24, 28], [21, 12, 41, 46, 2], [6, 35, 4, 30, 12], [20, 11, 3, 6, 24], [28, 30, 12, 41, 49], [3, 8, 35, 29, 21], [12, 20, 37, 5, 8], [35, 36, 21, 12, 20], [49, 3, 6, 25, 42], [30, 15, 17, 37, 5], [8, 32, 4, 30, 15], [41, 1, 0, 6, 13], [9, 27, 15, 41, 19], [5, 6, 35, 10, 40], [12, 41, 37, 0, 6], [25, 18, 27, 15, 20], [1, 5, 8, 13, 10], [39, 15, 17, 34, 5], [6, 25, 28, 27, 12], [17, 46, 0, 6, 13], [4, 23, 15, 20, 49], [0, 6, 24, 28, 27], [15, 17, 19, 3, 6], [13, 9, 21, 12, 17], [11, 2, 8, 25, 10], [39, 12, 41, 11, 5], [8, 24, 31, 21, 15], [26, 48, 7, 6, 35], [18, 40, 15, 38, 47], [3, 8, 35, 10, 21], [15, 20, 11, 5, 8], [24, 10, 40, 12, 20], [43, 2, 6, 32, 10], [21, 15, 41, 45, 0], [8, 35, 14, 22, 15], [20, 43, 5, 6, 35], [9, 23, 15, 17, 44], [0, 6, 25, 14, 21], [15, 17, 34, 7, 8], [24, 18, 23, 12, 20], [19, 5, 8, 13, 18], [27, 12, 38, 46, 5], [8, 35, 36, 27, 12], [17, 49, 7, 6, 32], [10, 30, 12, 26, 11], [0, 8, 25, 33, 27], [12, 20, 37, 5, 6], [32, 4, 21, 15, 17], [11, 2, 8, 24, 9], [30, 12, 41, 47, 7], [6, 16, 18, 27, 12], [20, 47, 7, 8, 16], [4, 27, 15, 26, 34], [2, 8, 13, 42, 30], [12, 26, 48, 7, 6], [24, 10, 27, 15, 41], [43, 5, 6, 16, 33], [27, 15, 38, 34, 0], [8, 25, 29, 39, 12], [17, 1, 7, 6, 35], [33, 23, 12, 26, 45], [0, 6, 25, 18, 22], [15, 20, 11, 0, 6], [16, 4, 39, 15, 17], [45, 2, 8, 16, 4], [21, 12, 17, 34, 3], [8, 35, 42, 23, 12], [20, 44, 3, 6, 25], [4, 40, 15, 41, 1], [2, 8, 24, 36, 30], [15, 38, 1, 3, 6], [35, 18, 22, 12, 26], [19, 2, 6, 13, 36], [39, 15, 20, 19, 7], [6, 32, 9, 27, 12], [26, 37, 2, 8, 16], [33, 40, 12, 26, 46], [3, 8, 35, 18, 39], [15, 26, 37, 0, 6], [16, 9, 39, 12, 38], [37, 7, 6, 25, 36], [21, 12, 41, 46, 5], [8, 24, 9, 40, 15], [38, 45, 0, 6, 32], [36, 27, 12, 41, 34], [0, 6, 16, 10, 21], [12, 26, 44, 5, 8], [16, 31, 21, 15, 17], [45, 0, 6, 35, 9], [30, 15, 17, 11, 3], [6, 24, 33, 39, 12], [20, 19, 0, 8, 35], [28, 30, 15, 20, 49], [2, 6, 24, 18, 27], [15, 38, 34, 5, 8], [16, 18, 27, 15, 38], [34, 7, 8, 35, 28], [39, 12, 17, 46, 3], [8, 13, 18, 39, 15], [17, 1, 2, 6, 35], [29, 30, 12, 20, 43], [7, 6, 25, 10, 27], [15, 20, 49, 5, 6], [24, 29, 23, 12, 38], [49, 7, 8, 35, 14], [22, 15, 38, 19, 5], [6, 16, 36, 23, 12], [38, 34, 7, 6, 32], [33, 40, 12, 41, 1], [0, 6, 35, 10, 40], [15, 17, 43, 0, 6], [16, 31, 30, 12, 17], [43, 5, 8, 24, 9], [23, 15, 41, 47, 7], [6, 35, 18, 22, 12], [26, 11, 5, 8, 24], [42, 40, 12, 26, 37], [2, 6, 25, 31, 40], [12, 41, 46, 2, 8], [25, 28, 30, 15, 26], [37, 2, 6, 35, 42], [23, 15, 41, 46, 5], [8, 24, 10, 21, 15], [26, 49, 2, 6, 35], [18, 23, 12, 26, 37], [2, 6, 24, 33, 22], [12, 41, 45, 7, 6], [24, 14, 23, 12, 26], [44, 5, 8, 35, 29], [21, 12, 20, 45, 3], [8, 13, 33, 40, 12], [20, 19, 0, 6, 13], [10, 27, 15, 17, 11], [5, 8, 13, 9, 39], [15, 41, 1, 3, 6], [13, 28, 30, 15, 17], [1, 3, 8, 13, 33], [40, 12, 26, 37, 7], [6, 13, 18, 23, 12], [20, 1, 7, 8, 24], [29, 27, 12, 26, 45], [2, 6, 32, 31, 22], [15, 38, 48, 0, 6], [24, 9, 21, 15, 17], [45, 3, 6, 35, 10], [27, 15, 41, 45, 2], [8, 32, 29, 39, 15], [38, 1, 7, 8, 13], [9, 39, 12, 17, 49], [0, 6, 32, 18, 27], [12, 41, 34, 5, 6], [16, 31, 21, 15, 26], [19, 5, 6, 13, 28], [22, 12, 26, 34, 5], [8, 32, 28, 39, 15], [41, 44, 7, 8, 25], [10, 40, 12, 41, 46], [0, 6, 24, 18, 39], [12, 17, 44, 7, 6], [24, 31, 30, 15, 41], [46, 3, 8, 32, 10], [39, 15, 26, 46, 2], [6, 16, 4, 27, 15], [17, 48, 3, 6, 35], [33, 30, 12, 20, 47], [7, 6, 24, 14, 21], [15, 38, 19, 3, 8], [13, 4, 22, 15, 26], [45, 0, 6, 32, 36], [39, 15, 38, 37, 7], [8, 32, 42, 30, 15], [38, 11, 5, 6, 35], [36, 27, 15, 26, 44], [2, 8, 16, 10, 40], [15, 20, 34, 2, 8], [13, 9, 23, 15, 17], [48, 3, 8, 32, 14], [22, 15, 38, 19, 7], [6, 24, 31, 27, 12], [41, 1, 0, 6, 13], [10, 40, 15, 20, 47], [5, 6, 24, 4, 30], [15, 26, 44, 7, 6], [35, 4, 30, 15, 26], [45, 5, 6, 32, 9], [30, 15, 20, 1, 7], [8, 35, 29, 22, 15], [41, 48, 3, 8, 16], [36, 40, 12, 20, 11], [2, 6, 16, 28, 21], [15, 38, 34, 2, 6], [35, 28, 30, 12, 38], [49, 2, 6, 24, 36], [39, 15, 38, 11, 0], [6, 16, 36, 23, 12], [26, 44, 3, 6, 35], [14, 23, 15, 41, 44], [0, 8, 16, 33, 22], [12, 20, 48, 0, 6], [16, 9, 39, 12, 17], [45, 2, 8, 25, 10], [21, 12, 17, 19, 7], [8, 13, 10, 30, 15], [17, 44, 2, 6, 16], [9, 21, 12, 17, 43], [7, 8, 24, 31, 27], [15, 38, 48, 0, 6], [24, 31, 21, 15, 20], [46, 2, 6, 25, 18], [21, 12, 20, 48, 2], [8, 16, 4, 27, 15], [41, 48, 5, 8, 13], [18, 39, 15, 41, 43], [2, 6, 32, 31, 22], [12, 17, 46, 7, 8], [32, 18, 21, 12, 26], [11, 0, 6, 35, 33], [27, 15, 17, 44, 7], [8, 25, 18, 27, 15], [17, 47, 5, 6, 24], [29, 22, 12, 41, 45], [0, 6, 32, 18, 23], [12, 26, 45, 2, 6], [35, 18, 23, 15, 26], [19, 0, 6, 16, 29], [30, 15, 20, 44, 7], [8, 16, 28, 21, 12], [20, 45, 2, 6, 16], [4, 22, 12, 38, 43], [7, 6, 25, 14, 39], [12, 20, 19, 7, 6], [24, 42, 27, 12, 26], [1, 7, 8, 25, 29], [23, 12, 41, 44, 5], [6, 25, 14, 22, 15], [26, 43, 5, 8, 16], [14, 27, 12, 41, 1], [0, 6, 24, 36, 22], [12, 41, 47, 2, 8], [24, 36, 39, 12, 17], [34, 2, 6, 24, 18], [39, 12, 38, 34, 3], [8, 25, 29, 27, 15], [41, 46, 5, 6, 35], [4, 30, 12, 41, 34], [0, 8, 35, 4, 39], [15, 26, 48, 3, 6], [24, 42, 23, 15, 38], [37, 5, 8, 24, 18], [39, 15, 38, 48, 7], [8, 35, 36, 30, 15], [20, 34, 2, 6, 25], [14, 23, 15, 26, 37], [0, 6, 13, 42, 23], [12, 41, 45, 3, 6], [35, 36, 27, 15, 17], [34, 3, 8, 24, 28], [27, 15, 20, 48, 2], [8, 25, 9, 21, 12], [20, 37, 0, 8, 24], [29, 40, 12, 41, 11], [7, 8, 24, 28, 40], [15, 20, 45, 5, 6], [35, 9, 22, 12, 20], [48, 2, 8, 16, 36], [30, 12, 41, 49, 3], [6, 35, 28, 22, 15], [26, 34, 2, 6, 35], [42, 40, 15, 41, 45], [3, 6, 32, 31, 40], [15, 41, 45, 2, 8], [24, 14, 39, 12, 26], [11, 3, 6, 16, 42], [23, 15, 26, 45, 5], [8, 13, 10, 30, 15], [20, 34, 7, 6, 16], [28, 39, 15, 38, 37], [7, 6, 24, 28, 39], [15, 41, 11, 5, 8], [16, 9, 21, 12, 17], [37, 2, 6, 16, 31], [39, 15, 38, 11, 7], [8, 16, 42, 39, 12], [17, 46, 3, 6, 13], [4, 30, 12, 41, 47], [0, 8, 25, 10, 30], [12, 41, 19, 2, 8], [32, 33, 22, 15, 41], [11, 3, 8, 24, 10], [40, 12, 26, 45, 0], [8, 35, 4, 23, 15], [17, 44, 3, 8, 35], [18, 40, 15, 38, 46], [3, 6, 13, 18, 23], [15, 17, 46, 3, 6], [16, 18, 22, 12, 17], [48, 5, 8, 13, 31], [30, 12, 20, 46, 0], [6, 13, 42, 23, 15], [20, 47, 0, 8, 16], [31, 40, 15, 41, 11], [3, 8, 16, 28, 30], [15, 17, 19, 7, 6], [24, 36, 39, 15, 20], [11, 5, 6, 35, 10], [40, 12, 17, 49, 3], [6, 32, 33, 27, 12], [17, 34, 2, 8, 24], [9, 22, 12, 20, 46], [2, 6, 35, 9, 39], [15, 26, 49, 5, 6], [13, 33, 23, 12, 17], [45, 0, 6, 16, 42], [27, 12, 20, 47, 3], [6, 32, 14, 39, 12], [38, 11, 5, 8, 32], [42, 39, 15, 17, 1], [2, 6, 35, 42, 40], [15, 26, 47, 5, 8], [24, 10, 27, 12, 38], [37, 2, 6, 25, 18], [30, 12, 20, 48, 0], [8, 16, 29, 27, 15], [17, 46, 3, 8, 32], [42, 21, 15, 20, 37], [7, 8, 16, 10, 40], [15, 17, 49, 0, 6], [24, 18, 27, 12, 41], [44, 2, 6, 16, 31], [22, 12, 41, 44, 2], [6, 13, 33, 30, 15], [26, 44, 2, 8, 13], [28, 23, 12, 17, 19], [5, 8, 16, 33, 21], [12, 26, 49, 3, 6]]
        if how_many_parts == 20:
            picked_model = [[25, 33, 40, 15, 41, 37, 0, 8, 25, 33], [23, 12, 17, 48, 7, 6, 13, 28, 27, 15], [41, 19, 7, 6, 35, 42, 40, 12, 20, 11], [5, 8, 16, 42, 39, 12, 38, 34, 5, 8], [35, 31, 27, 15, 20, 11, 7, 6, 35, 42], [23, 12, 20, 19, 7, 8, 25, 9, 30, 15], [41, 34, 3, 6, 24, 42, 40, 15, 41, 1], [2, 6, 24, 31, 22, 15, 26, 45, 2, 6], [35, 31, 22, 15, 26, 43, 0, 6, 25, 4], [23, 12, 41, 37, 0, 8, 13, 36, 40, 15], [38, 19, 7, 6, 24, 9, 40, 15, 17, 19], [7, 6, 25, 4, 39, 12, 20, 11, 5, 8], [16, 4, 39, 12, 26, 43, 5, 6, 13, 31], [23, 12, 17, 11, 5, 6, 32, 18, 23, 12], [26, 37, 0, 8, 25, 36, 27, 15, 20, 48], [7, 6, 16, 33, 23, 15, 38, 44, 5, 6], [13, 10, 22, 12, 41, 43, 2, 8, 35, 33], [30, 12, 17, 45, 7, 6, 25, 36, 21, 12], [38, 48, 0, 6, 16, 18, 40, 15, 26, 11], [5, 6, 16, 14, 40, 12, 38, 47, 0, 6], [13, 28, 40, 12, 20, 43, 5, 8, 25, 28], [23, 12, 41, 11, 0, 6, 35, 42, 23, 15], [26, 48, 2, 6, 25, 42, 40, 15, 26, 19], [7, 6, 16, 28, 22, 15, 20, 46, 7, 8], [35, 31, 40, 12, 20, 1, 7, 6, 13, 36], [23, 15, 20, 11, 3, 8, 13, 28, 27, 15], [41, 37, 0, 6, 13, 10, 39, 15, 20, 43], [0, 6, 35, 28, 22, 15, 38, 44, 7, 6], [35, 14, 21, 12, 41, 48, 2, 8, 24, 9], [23, 12, 17, 46, 2, 6, 25, 28, 40, 12], [20, 34, 5, 6, 13, 33, 23, 12, 26, 1], [2, 8, 25, 36, 27, 15, 26, 19, 3, 8], [24, 36, 27, 15, 20, 19, 5, 8, 32, 4], [22, 12, 38, 37, 5, 6, 35, 18, 27, 15], [17, 44, 5, 6, 35, 31, 39, 15, 26, 11], [2, 8, 13, 33, 27, 12, 26, 48, 7, 8], [16, 29, 21, 15, 20, 46, 5, 8, 24, 9], [30, 12, 26, 37, 2, 8, 24, 36, 22, 12], [17, 37, 3, 6, 25, 36, 27, 15, 38, 1], [7, 8, 16, 14, 39, 15, 41, 48, 0, 6], [13, 14, 39, 15, 26, 45, 5, 8, 35, 14], [23, 12, 26, 1, 0, 6, 24, 9, 27, 15], [38, 37, 7, 8, 16, 33, 22, 12, 26, 1], [0, 6, 24, 42, 30, 12, 38, 48, 2, 6], [25, 9, 22, 15, 17, 37, 2, 6, 13, 36], [39, 12, 41, 37, 3, 6, 24, 29, 30, 15], [41, 1, 3, 8, 13, 28, 30, 15, 17, 43], [5, 8, 25, 18, 30, 12, 26, 37, 5, 8], [24, 33, 27, 12, 26, 49, 3, 8, 16, 10], [22, 12, 41, 1, 5, 6, 24, 28, 22, 15], [17, 19, 3, 8, 16, 9, 27, 12, 20, 11], [5, 8, 35, 29, 21, 15, 20, 45, 0, 6], [35, 10, 23, 12, 20, 43, 0, 8, 35, 10], [39, 15, 20, 11, 3, 8, 25, 29, 30, 12], [38, 1, 5, 8, 13, 36, 40, 15, 41, 1], [5, 8, 16, 9, 21, 12, 26, 48, 3, 6], [35, 36, 21, 15, 41, 48, 7, 6, 25, 18], [21, 15, 38, 37, 7, 6, 13, 9, 22, 15], [38, 49, 0, 6, 24, 9, 40, 15, 20, 44], [0, 6, 13, 4, 23, 15, 17, 45, 5, 6], [13, 36, 22, 12, 20, 43, 5, 8, 16, 31], [21, 12, 17, 43, 5, 8, 24, 10, 39, 12], [26, 1, 7, 8, 35, 4, 23, 12, 41, 43], [5, 6, 32, 18, 22, 12, 20, 43, 2, 6], [24, 29, 40, 12, 38, 46, 2, 6, 24, 42], [23, 12, 26, 47, 5, 6, 35, 36, 21, 12], [17, 43, 7, 8, 24, 33, 22, 15, 17, 48], [7, 6, 13, 18, 21, 15, 17, 47, 2, 6], [24, 18, 30, 15, 41, 11, 0, 6, 24, 31], [23, 15, 20, 44, 5, 8, 35, 14, 30, 15], [26, 46, 7, 8, 13, 14, 40, 12, 17, 11], [0, 8, 32, 18, 27, 12, 17, 43, 7, 6], [35, 31, 40, 15, 38, 11, 7, 8, 32, 31], [21, 15, 17, 48, 7, 6, 16, 4, 40, 15], [38, 45, 2, 6, 24, 29, 22, 15, 20, 46], [7, 6, 16, 9, 30, 15, 38, 1, 5, 6], [32, 14, 21, 12, 17, 47, 0, 8, 24, 42], [30, 15, 17, 46, 7, 8, 24, 29, 22, 12], [17, 44, 2, 8, 35, 18, 27, 12, 41, 34], [0, 8, 24, 29, 39, 12, 26, 19, 3, 8], [25, 10, 30, 12, 20, 11, 3, 6, 24, 42], [23, 12, 26, 11, 2, 8, 25, 28, 22, 15], [41, 44, 2, 8, 13, 33, 40, 12, 17, 43], [0, 8, 24, 18, 30, 12, 20, 49, 7, 8], [35, 4, 23, 15, 17, 11, 2, 6, 25, 28], [27, 12, 26, 43, 0, 8, 32, 9, 27, 15], [20, 46, 0, 8, 35, 31, 21, 12, 38, 19], [5, 8, 35, 36, 22, 12, 17, 43, 3, 6], [24, 18, 40, 15, 41, 1, 2, 6, 24, 42], [40, 15, 26, 19, 7, 6, 32, 31, 21, 15], [26, 1, 3, 6, 16, 33, 22, 12, 41, 1], [0, 6, 13, 36, 40, 12, 41, 44, 7, 6], [25, 4, 22, 12, 41, 37, 2, 8, 16, 28], [21, 12, 41, 11, 5, 8, 32, 33, 23, 12], [17, 49, 7, 8, 13, 18, 23, 12, 17, 49], [5, 6, 16, 36, 21, 12, 20, 45, 3, 6], [25, 29, 39, 12, 17, 47, 7, 8, 16, 28], [22, 12, 20, 45, 0, 8, 25, 42, 39, 15], [26, 19, 3, 6, 32, 36, 40, 12, 41, 37], [0, 6, 35, 9, 30, 12, 17, 48, 3, 6], [32, 10, 27, 12, 38, 11, 7, 6, 35, 31], [22, 12, 17, 43, 7, 8, 16, 9, 39, 15], [20, 44, 3, 8, 16, 18, 22, 15, 38, 43], [2, 8, 32, 9, 21, 15, 38, 34, 3, 6], [32, 31, 21, 12, 41, 11, 5, 8, 25, 31], [23, 15, 20, 1, 0, 6, 13, 9, 22, 15], [20, 45, 2, 6, 32, 42, 40, 15, 38, 43], [5, 6, 16, 29, 40, 15, 26, 1, 2, 6], [16, 28, 40, 12, 26, 1, 7, 8, 24, 36], [21, 12, 38, 44, 0, 8, 13, 14, 39, 15], [20, 34, 3, 8, 24, 31, 23, 15, 26, 49], [2, 8, 16, 42, 30, 12, 17, 49, 3, 8], [35, 9, 22, 15, 17, 1, 0, 8, 35, 36], [22, 12, 41, 45, 0, 6, 13, 4, 39, 15], [17, 44, 7, 6, 35, 42, 39, 15, 20, 1], [7, 8, 35, 42, 23, 12, 38, 44, 0, 8], [13, 42, 39, 12, 38, 1, 7, 8, 32, 10], [23, 12, 17, 48, 7, 6, 32, 33, 30, 12], [20, 47, 2, 6, 16, 9, 23, 12, 38, 1], [3, 8, 35, 4, 40, 12, 41, 19, 5, 8], [16, 36, 22, 12, 26, 11, 3, 8, 24, 10], [40, 15, 41, 1, 7, 6, 16, 36, 39, 12], [26, 47, 3, 8, 13, 36, 27, 15, 38, 1], [5, 8, 25, 33, 21, 12, 20, 44, 2, 6], [25, 28, 39, 12, 26, 46, 2, 8, 32, 4], [39, 12, 26, 1, 7, 8, 24, 9, 30, 12], [26, 48, 0, 6, 24, 18, 27, 15, 41, 46], [5, 8, 35, 10, 30, 12, 20, 37, 7, 6], [13, 36, 22, 12, 38, 34, 3, 6, 16, 18], [22, 15, 17, 48, 7, 6, 32, 33, 39, 12], [38, 19, 5, 6, 32, 10, 40, 12, 26, 49], [0, 6, 35, 42, 30, 15, 41, 46, 3, 8], [35, 18, 27, 15, 17, 46, 0, 8, 32, 31], [40, 15, 20, 34, 0, 6, 13, 33, 27, 12], [26, 34, 3, 8, 13, 14, 39, 12, 41, 43], [7, 6, 35, 18, 27, 15, 20, 1, 7, 6], [35, 36, 27, 12, 17, 43, 3, 6, 24, 10], [30, 12, 17, 1, 5, 6, 24, 18, 30, 12], [38, 49, 7, 6, 13, 9, 39, 12, 38, 46], [0, 6, 35, 29, 40, 15, 41, 48, 7, 6], [16, 9, 21, 12, 20, 44, 5, 6, 25, 18], [21, 15, 26, 43, 3, 6, 32, 31, 30, 15], [26, 44, 5, 8, 35, 31, 40, 15, 26, 1], [5, 8, 13, 29, 39, 12, 17, 48, 7, 8], [24, 36, 21, 15, 20, 49, 2, 8, 32, 4], [27, 12, 26, 1, 7, 6, 35, 18, 22, 15], [41, 34, 7, 8, 16, 36, 39, 12, 20, 37], [0, 6, 16, 4, 40, 12, 20, 48, 5, 6], [13, 42, 27, 12, 20, 11, 5, 6, 16, 29], [30, 15, 41, 1, 5, 6, 25, 18, 21, 12], [26, 11, 7, 8, 24, 4, 39, 12, 20, 47], [2, 6, 24, 33, 40, 12, 17, 47, 7, 6], [35, 18, 30, 15, 26, 1, 7, 6, 35, 14], [39, 12, 38, 19, 5, 6, 25, 10, 39, 12], [17, 43, 0, 6, 24, 9, 23, 15, 17, 19], [7, 8, 13, 14, 23, 15, 41, 11, 3, 6], [32, 36, 27, 15, 38, 37, 5, 6, 13, 28], [21, 15, 17, 45, 2, 8, 25, 31, 22, 15], [26, 1, 3, 6, 32, 18, 30, 15, 17, 37], [0, 8, 24, 33, 22, 12, 41, 46, 3, 8], [35, 42, 21, 12, 17, 43, 5, 6, 35, 28], [40, 12, 41, 48, 3, 8, 13, 14, 39, 15], [41, 46, 7, 8, 25, 14, 22, 15, 17, 1], [5, 6, 32, 33, 21, 12, 17, 49, 3, 8], [16, 10, 27, 15, 20, 37, 2, 6, 24, 31], [21, 15, 26, 47, 0, 6, 25, 9, 23, 15], [26, 45, 3, 8, 35, 4, 21, 12, 26, 48], [5, 8, 24, 36, 27, 12, 41, 45, 5, 6], [32, 33, 22, 12, 17, 47, 7, 6, 35, 36], [23, 15, 41, 1, 2, 8, 13, 29, 40, 12], [26, 19, 7, 8, 32, 10, 23, 15, 26, 44], [3, 8, 35, 14, 40, 15, 20, 34, 2, 6], [32, 33, 40, 15, 17, 45, 0, 6, 24, 14], [21, 12, 41, 46, 7, 6, 32, 31, 23, 15], [17, 37, 7, 8, 35, 31, 39, 12, 26, 46], [2, 8, 35, 33, 40, 12, 38, 49, 2, 8], [24, 18, 39, 15, 41, 49, 0, 8, 25, 18], [40, 12, 41, 43, 0, 6, 16, 28, 23, 15], [38, 37, 0, 8, 32, 28, 23, 12, 20, 11], [0, 6, 32, 42, 21, 15, 38, 44, 5, 8], [13, 18, 23, 12, 38, 48, 5, 6, 24, 28], [22, 15, 20, 37, 2, 8, 32, 42, 39, 15], [20, 43, 0, 8, 16, 18, 30, 12, 17, 34], [7, 6, 35, 4, 27, 12, 38, 11, 2, 6], [32, 28, 39, 15, 26, 47, 3, 8, 32, 42], [21, 15, 41, 43, 2, 8, 35, 28, 21, 15], [17, 1, 7, 6, 13, 29, 21, 15, 41, 1], [0, 8, 32, 29, 27, 15, 20, 11, 0, 8], [25, 33, 27, 15, 20, 49, 2, 8, 35, 18], [30, 12, 26, 19, 3, 8, 24, 10, 27, 12], [41, 44, 2, 8, 32, 33, 40, 15, 38, 1], [0, 8, 13, 42, 30, 12, 38, 49, 7, 6], [25, 28, 27, 12, 41, 47, 3, 8, 13, 29], [40, 12, 17, 44, 3, 8, 16, 18, 30, 12], [17, 1, 0, 6, 13, 33, 22, 15, 38, 44], [7, 8, 16, 28, 21, 12, 38, 43, 0, 8], [32, 28, 40, 12, 20, 48, 7, 6, 32, 33], [23, 15, 26, 19, 3, 6, 16, 33, 22, 12], [17, 11, 0, 8, 24, 14, 39, 12, 17, 44], [5, 8, 24, 14, 40, 12, 38, 44, 3, 6], [13, 42, 21, 15, 20, 37, 2, 8, 35, 36], [22, 12, 41, 34, 0, 6, 25, 31, 27, 15], [38, 47, 7, 8, 32, 31, 27, 15, 26, 1], [3, 6, 13, 18, 22, 12, 17, 43, 3, 8], [25, 28, 22, 15, 26, 45, 5, 8, 24, 4], [40, 12, 20, 44, 3, 8, 16, 18, 30, 15], [17, 44, 2, 6, 13, 29, 30, 15, 26, 11], [3, 6, 13, 29, 30, 12, 17, 44, 0, 6], [25, 18, 22, 15, 20, 48, 3, 6, 24, 28], [22, 15, 41, 11, 7, 8, 35, 10, 22, 12], [26, 11, 3, 6, 32, 31, 27, 12, 17, 11], [7, 8, 13, 33, 21, 12, 20, 44, 0, 6], [16, 36, 27, 12, 38, 11, 7, 6, 35, 31], [27, 12, 20, 47, 2, 6, 25, 36, 39, 15], [20, 11, 0, 8, 32, 9, 40, 12, 26, 37], [2, 6, 32, 14, 23, 15, 41, 49, 2, 8], [35, 9, 27, 15, 38, 34, 7, 6, 13, 36], [27, 12, 26, 19, 0, 6, 13, 10, 27, 15], [20, 43, 3, 6, 25, 4, 40, 12, 20, 48], [7, 6, 25, 4, 21, 15, 26, 49, 2, 8], [32, 28, 23, 15, 41, 34, 3, 8, 32, 33], [21, 12, 38, 48, 3, 8, 32, 31, 39, 12], [17, 43, 5, 6, 25, 28, 21, 15, 26, 48], [2, 8, 32, 42, 23, 15, 26, 37, 3, 8], [25, 18, 40, 12, 20, 47, 2, 6, 35, 42], [27, 12, 26, 44, 7, 8, 13, 33, 27, 15], [17, 49, 7, 8, 16, 10, 23, 15, 41, 11], [7, 8, 35, 31, 39, 12, 20, 45, 2, 8], [13, 31, 23, 15, 20, 19, 2, 8, 24, 4], [23, 12, 41, 48, 7, 6, 16, 31, 40, 12], [41, 48, 7, 8, 35, 36, 40, 12, 38, 49], [2, 8, 24, 18, 22, 15, 17, 19, 0, 6], [24, 9, 22, 15, 17, 49, 3, 6, 35, 36], [30, 12, 20, 1, 2, 8, 32, 10, 30, 15], [20, 37, 3, 8, 25, 28, 30, 15, 20, 44], [3, 8, 25, 33, 23, 15, 26, 49, 0, 8], [25, 36, 23, 12, 26, 11, 0, 6, 32, 4], [27, 12, 17, 37, 3, 6, 25, 31, 21, 15], [41, 47, 0, 6, 13, 33, 30, 15, 20, 47], [7, 6, 35, 42, 23, 15, 41, 34, 3, 8], [32, 31, 21, 12, 41, 48, 5, 8, 16, 9], [22, 15, 26, 45, 5, 8, 13, 31, 27, 15], [26, 47, 3, 8, 32, 29, 21, 12, 17, 48], [5, 6, 25, 33, 21, 15, 20, 43, 2, 8], [25, 4, 39, 15, 26, 37, 5, 8, 13, 10], [21, 12, 20, 19, 5, 6, 24, 10, 27, 15], [17, 19, 5, 8, 16, 33, 21, 15, 41, 19], [5, 8, 35, 18, 23, 15, 17, 44, 7, 8], [35, 31, 30, 15, 41, 37, 3, 8, 25, 33], [30, 15, 41, 49, 0, 6, 35, 36, 23, 12], [20, 49, 2, 8, 25, 14, 30, 15, 20, 49], [7, 8, 24, 42, 30, 15, 38, 49, 2, 6], [24, 31, 22, 12, 17, 46, 3, 8, 24, 4], [30, 15, 38, 46, 5, 6, 24, 18, 39, 15], [38, 46, 0, 8, 25, 31, 22, 15, 38, 43], [2, 8, 25, 14, 39, 12, 26, 11, 7, 8], [24, 14, 21, 15, 17, 11, 7, 6, 35, 9], [27, 15, 20, 47, 7, 8, 13, 33, 30, 15], [20, 19, 0, 8, 24, 31, 30, 15, 41, 44], [3, 8, 32, 10, 39, 15, 26, 1, 2, 8], [13, 18, 23, 12, 17, 49, 5, 6, 35, 9], [22, 12, 41, 45, 7, 8, 35, 14, 40, 12], [20, 45, 5, 6, 32, 10, 40, 12, 38, 46], [7, 8, 13, 36, 21, 15, 38, 49, 0, 6], [25, 42, 40, 15, 20, 11, 3, 8, 32, 9], [21, 15, 41, 48, 0, 6, 13, 9, 21, 15], [38, 34, 3, 6, 32, 36, 30, 12, 20, 1], [0, 8, 35, 33, 30, 12, 26, 47, 2, 8], [32, 29, 22, 15, 20, 1, 5, 8, 24, 28], [27, 15, 26, 45, 7, 8, 25, 31, 23, 15], [38, 19, 7, 6, 35, 14, 22, 12, 20, 49], [0, 6, 16, 9, 39, 12, 20, 34, 3, 8], [25, 28, 40, 15, 20, 19, 5, 6, 25, 33], [27, 12, 26, 49, 0, 8, 24, 14, 21, 12], [26, 49, 2, 8, 35, 36, 40, 15, 26, 19], [3, 8, 13, 42, 39, 12, 20, 11, 5, 6], [32, 31, 40, 15, 38, 45, 3, 8, 16, 14], [22, 12, 17, 19, 7, 6, 13, 4, 22, 15], [41, 19, 3, 6, 32, 14, 30, 15, 41, 19], [7, 6, 32, 29, 27, 15, 17, 46, 5, 6], [16, 28, 40, 12, 38, 19, 7, 6, 35, 36], [39, 12, 17, 1, 7, 6, 35, 31, 23, 12], [26, 19, 5, 8, 35, 9, 23, 12, 41, 44], [3, 6, 13, 42, 21, 15, 20, 46, 5, 6], [16, 33, 40, 12, 20, 34, 3, 8, 16, 10], [39, 15, 41, 1, 3, 6, 24, 29, 39, 12], [41, 46, 2, 8, 32, 28, 22, 15, 26, 37], [3, 6, 13, 36, 22, 12, 26, 47, 0, 8], [32, 36, 30, 15, 20, 1, 2, 8, 35, 29], [23, 15, 17, 43, 3, 6, 25, 14, 21, 15], [26, 47, 0, 6, 32, 18, 30, 15, 26, 46], [0, 6, 13, 10, 22, 12, 20, 19, 5, 6], [13, 10, 27, 15, 20, 44, 5, 8, 24, 36], [22, 12, 41, 11, 0, 8, 13, 29, 40, 15], [41, 46, 0, 8, 32, 28, 22, 15, 38, 49], [5, 8, 24, 10, 23, 15, 20, 47, 3, 8], [13, 18, 22, 15, 38, 1, 3, 6, 24, 36], [40, 15, 17, 1, 0, 8, 24, 4, 39, 12], [26, 11, 7, 6, 24, 14, 30, 15, 41, 48], [5, 8, 13, 4, 23, 12, 38, 44, 0, 6], [25, 42, 21, 15, 20, 49, 0, 6, 24, 9], [39, 15, 17, 44, 3, 8, 35, 18, 23, 15], [17, 34, 7, 8, 32, 31, 23, 12, 38, 49], [7, 8, 13, 4, 40, 12, 38, 34, 3, 8], [25, 33, 39, 15, 26, 47, 0, 6, 16, 28], [23, 12, 26, 46, 0, 6, 25, 33, 39, 15], [41, 43, 5, 8, 16, 10, 22, 12, 26, 11], [3, 6, 35, 36, 39, 12, 20, 1, 7, 8], [16, 10, 22, 15, 17, 19, 3, 8, 35, 29], [22, 12, 26, 19, 5, 8, 13, 42, 23, 15], [20, 34, 2, 6, 25, 42, 22, 12, 41, 44], [0, 6, 25, 31, 39, 12, 26, 19, 3, 8], [13, 14, 40, 12, 20, 19, 0, 8, 32, 14], [23, 12, 41, 46, 5, 6, 32, 29, 21, 12], [26, 47, 2, 8, 35, 29, 22, 15, 20, 37], [7, 6, 25, 10, 40, 15, 20, 44, 5, 6], [16, 14, 27, 12, 17, 37, 5, 6, 25, 9], [39, 15, 41, 43, 7, 6, 16, 18, 30, 15], [26, 48, 5, 8, 32, 29, 21, 15, 17, 47], [5, 8, 16, 10, 27, 15, 17, 37, 2, 8], [24, 42, 30, 15, 41, 34, 0, 6, 35, 9], [21, 15, 41, 19, 2, 6, 24, 36, 27, 15], [38, 19, 0, 6, 24, 28, 23, 12, 20, 49], [2, 6, 35, 33, 27, 15, 26, 45, 0, 6], [25, 29, 21, 12, 17, 45, 5, 8, 25, 31], [40, 12, 20, 48, 7, 6, 25, 31, 22, 12], [26, 43, 0, 8, 32, 36, 27, 15, 41, 1], [3, 8, 32, 29, 21, 15, 41, 37, 0, 8], [24, 42, 27, 15, 17, 44, 2, 6, 32, 31], [27, 12, 17, 34, 2, 8, 32, 36, 30, 12], [20, 45, 5, 6, 13, 31, 23, 12, 17, 1], [3, 8, 24, 31, 21, 12, 41, 44, 5, 6], [32, 36, 21, 12, 17, 49, 5, 8, 32, 29], [40, 15, 41, 1, 7, 8, 25, 29, 30, 12], [17, 45, 3, 8, 35, 36, 23, 12, 38, 1], [0, 6, 25, 31, 23, 15, 38, 1, 7, 6], [35, 9, 23, 12, 41, 45, 2, 6, 25, 4], [22, 15, 17, 47, 2, 8, 13, 9, 40, 15], [38, 43, 3, 8, 35, 36, 21, 15, 41, 44], [5, 8, 16, 29, 27, 15, 41, 11, 3, 8], [35, 33, 23, 12, 41, 1, 3, 8, 16, 10], [21, 15, 17, 46, 3, 6, 16, 4, 22, 12], [38, 37, 3, 8, 16, 29, 40, 12, 26, 46], [2, 8, 32, 33, 23, 15, 17, 34, 2, 6], [13, 31, 30, 15, 17, 45, 0, 8, 32, 4], [30, 12, 41, 44, 0, 8, 35, 14, 40, 15], [17, 47, 2, 8, 32, 36, 40, 12, 38, 44], [0, 6, 16, 9, 21, 12, 20, 48, 3, 6], [25, 28, 22, 15, 41, 45, 2, 6, 16, 31], [27, 15, 20, 11, 2, 8, 13, 36, 39, 15], [17, 49, 0, 8, 32, 4, 40, 15, 20, 19], [0, 8, 25, 28, 40, 15, 17, 44, 2, 8], [16, 4, 23, 15, 26, 43, 2, 8, 13, 28], [23, 15, 41, 37, 2, 8, 25, 42, 39, 15], [20, 45, 2, 6, 24, 31, 23, 12, 41, 11], [2, 8, 32, 18, 30, 15, 20, 45, 7, 8], [16, 31, 30, 15, 20, 11, 5, 6, 25, 31], [27, 15, 26, 44, 3, 8, 35, 33, 22, 12], [26, 48, 3, 6, 35, 29, 23, 15, 41, 43], [7, 8, 32, 31, 27, 12, 26, 47, 3, 8], [16, 33, 30, 12, 26, 37, 0, 6, 35, 36], [30, 15, 20, 44, 0, 8, 32, 9, 21, 15], [41, 45, 3, 8, 35, 14, 21, 12, 38, 37], [2, 8, 24, 9, 23, 15, 20, 37, 3, 8], [16, 28, 30, 15, 38, 1, 7, 6, 24, 36], [39, 12, 20, 34, 5, 6, 25, 33, 30, 12], [20, 43, 3, 8, 35, 9, 40, 12, 20, 44], [3, 8, 13, 10, 27, 12, 38, 37, 5, 6], [25, 36, 27, 15, 38, 1, 0, 6, 16, 14], [30, 12, 17, 37, 5, 6, 24, 9, 30, 15], [17, 1, 5, 6, 13, 4, 22, 12, 38, 48], [5, 8, 24, 9, 27, 15, 26, 34, 5, 8], [13, 36, 40, 15, 41, 19, 5, 8, 24, 28], [30, 12, 20, 11, 0, 6, 32, 33, 27, 12], [20, 44, 3, 6, 13, 28, 40, 15, 38, 43], [5, 6, 35, 31, 30, 15, 17, 43, 7, 6], [13, 9, 21, 12, 41, 19, 5, 6, 24, 31], [40, 12, 20, 43, 0, 8, 13, 33, 40, 15], [38, 37, 2, 8, 13, 4, 23, 15, 41, 34], [2, 8, 32, 4, 30, 15, 41, 47, 3, 8], [32, 28, 22, 15, 26, 11, 5, 6, 25, 9], [23, 12, 38, 44, 0, 8, 25, 28, 27, 15], [26, 46, 5, 6, 25, 33, 39, 12, 17, 44], [7, 8, 32, 28, 40, 15, 20, 44, 7, 6], [24, 33, 22, 15, 38, 46, 5, 6, 16, 33], [39, 15, 17, 1, 2, 6, 16, 28, 22, 12], [17, 49, 3, 8, 35, 4, 40, 12, 17, 43], [5, 8, 25, 36, 22, 12, 20, 1, 3, 6], [13, 29, 23, 15, 20, 45, 5, 8, 13, 31], [27, 12, 38, 49, 2, 6, 13, 28, 30, 12], [26, 48, 0, 6, 32, 4, 40, 15, 38, 45], [5, 8, 13, 33, 39, 12, 41, 19, 2, 6], [35, 42, 22, 15, 38, 49, 7, 6, 16, 28], [22, 15, 20, 37, 7, 6, 35, 14, 39, 12], [17, 49, 2, 8, 32, 36, 30, 15, 17, 45], [5, 8, 24, 36, 23, 15, 26, 45, 2, 8], [24, 18, 39, 12, 17, 1, 0, 6, 13, 36], [23, 12, 17, 37, 3, 6, 25, 28, 40, 12], [38, 44, 0, 6, 16, 28, 30, 12, 17, 47], [5, 6, 24, 10, 40, 15, 41, 11, 3, 6]]
        if how_many_parts == 5:
            picked_model = [[24, 9, 22], [12, 17, 11], [3, 8, 35], [4, 39, 12], [26, 45, 5], [8, 16, 36], [30, 12, 20], [34, 2, 6], [13, 33, 39], [12, 26, 43], [7, 6, 32], [42, 27, 15], [38, 37, 3], [6, 35, 33], [39, 12, 17], [19, 2, 6], [24, 42, 21], [12, 38, 47], [0, 8, 25], [10, 40, 12], [38, 49, 7], [8, 25, 4], [30, 15, 38], [48, 3, 8], [24, 9, 27], [15, 17, 45], [0, 6, 25], [29, 22, 15], [17, 47, 3], [8, 24, 18], [40, 12, 17], [34, 7, 6], [35, 10, 22], [12, 17, 1], [0, 6, 16], [14, 21, 12], [26, 47, 7], [6, 35, 4], [30, 12, 38], [49, 3, 8], [13, 14, 21], [12, 26, 45], [2, 8, 25], [36, 21, 15], [26, 11, 2], [6, 32, 10], [22, 15, 26], [48, 3, 8], [32, 28, 30], [15, 38, 19], [5, 8, 24], [42, 40, 15], [41, 48, 3], [6, 24, 9], [39, 15, 41], [11, 3, 8], [13, 33, 27], [12, 17, 1], [7, 6, 13], [33, 30, 12], [20, 48, 5], [8, 24, 42], [27, 15, 26], [45, 0, 6], [25, 4, 30], [12, 41, 48], [2, 6, 24], [31, 21, 15], [41, 49, 2], [8, 16, 28], [23, 12, 20], [48, 7, 6], [13, 18, 30], [12, 17, 45], [0, 6, 25], [28, 23, 12], [20, 45, 3], [6, 32, 36], [23, 12, 38], [46, 3, 6], [24, 42, 30], [12, 17, 19], [3, 6, 24], [36, 21, 12], [26, 48, 3], [6, 24, 36], [21, 15, 20], [45, 2, 6], [25, 4, 27], [12, 20, 34], [2, 6, 13], [36, 39, 12], [41, 34, 7], [6, 32, 9], [27, 15, 20], [44, 3, 6], [16, 42, 30], [12, 26, 1], [7, 6, 25], [4, 22, 15], [38, 48, 7], [8, 25, 14], [22, 12, 26], [47, 2, 6], [16, 31, 21], [12, 38, 49], [5, 8, 24], [10, 23, 15], [17, 46, 0], [8, 35, 36], [23, 12, 17], [48, 3, 6], [13, 14, 30], [12, 20, 11], [2, 6, 35], [33, 30, 12], [20, 43, 0], [6, 16, 9], [40, 12, 41], [47, 0, 6], [13, 10, 21], [15, 41, 19], [3, 8, 16], [29, 30, 12], [38, 46, 3], [8, 13, 31], [30, 15, 26], [49, 3, 6], [32, 31, 22], [15, 41, 47], [5, 6, 32], [31, 39, 12], [26, 19, 7], [6, 25, 29], [30, 12, 26], [46, 5, 8], [35, 9, 27], [12, 41, 37], [3, 8, 32], [31, 21, 15], [41, 1, 0], [8, 35, 36], [39, 12, 26], [46, 7, 8], [25, 33, 40], [15, 20, 1], [5, 6, 24], [18, 22, 12], [17, 47, 0], [8, 32, 18], [22, 15, 20], [47, 0, 8], [25, 42, 30], [15, 38, 1], [7, 8, 32], [4, 40, 12], [26, 37, 7], [8, 25, 31], [21, 12, 41], [1, 2, 6], [24, 14, 40], [15, 41, 46], [0, 6, 13], [4, 30, 15], [38, 19, 7], [8, 32, 18], [22, 12, 20], [19, 0, 6], [32, 4, 40], [12, 17, 45], [0, 6, 25], [14, 40, 15], [26, 1, 3], [6, 16, 28], [21, 12, 17], [48, 7, 8], [32, 28, 30], [12, 41, 47], [2, 8, 25], [14, 40, 15], [26, 19, 3], [8, 13, 10], [30, 12, 17], [46, 5, 8], [24, 18, 40], [15, 41, 43], [7, 6, 35], [31, 39, 15], [38, 48, 0], [8, 16, 10], [39, 15, 41], [34, 2, 8], [13, 33, 27], [12, 17, 11], [0, 8, 24], [4, 22, 12], [38, 19, 2], [8, 35, 36], [22, 15, 17], [43, 3, 8], [16, 36, 40], [12, 26, 1], [0, 8, 32], [31, 27, 12], [20, 47, 5], [8, 24, 31], [27, 15, 20], [43, 2, 6], [32, 9, 30], [12, 26, 48], [7, 6, 32], [18, 39, 15], [20, 46, 3], [6, 32, 14], [27, 12, 17], [11, 2, 6], [13, 9, 21], [15, 41, 47], [5, 8, 16], [36, 30, 12], [26, 44, 5], [8, 24, 36], [21, 12, 38], [49, 0, 8], [32, 18, 40], [15, 20, 47], [3, 6, 24], [42, 40, 12], [26, 1, 2], [6, 25, 4], [27, 12, 17], [43, 7, 8], [35, 33, 39], [12, 38, 43], [2, 8, 24], [4, 30, 15], [26, 49, 5], [6, 32, 33], [22, 12, 26], [47, 7, 8], [25, 14, 40], [15, 26, 47], [3, 8, 13], [36, 21, 15], [26, 34, 7], [8, 13, 9], [39, 15, 17], [47, 7, 8], [24, 33, 22], [15, 20, 11], [7, 6, 35], [14, 30, 12], [26, 37, 5], [8, 24, 18], [40, 12, 17], [46, 5, 6], [13, 9, 39], [12, 41, 19], [0, 6, 24], [14, 21, 12], [20, 37, 0], [8, 16, 28], [22, 15, 20], [46, 0, 8], [13, 9, 40], [12, 38, 49], [3, 8, 24], [42, 21, 15], [26, 37, 2], [6, 35, 36], [39, 12, 26], [19, 0, 8], [13, 14, 21], [15, 26, 48], [2, 8, 32], [10, 23, 15], [20, 34, 5], [8, 13, 33], [30, 12, 20], [48, 0, 6], [35, 31, 40], [12, 17, 11], [0, 6, 32], [10, 22, 12], [17, 46, 3], [8, 13, 42], [22, 12, 17], [45, 7, 6], [32, 4, 40], [15, 17, 48], [3, 6, 32], [29, 30, 12], [20, 19, 0], [6, 24, 10], [39, 12, 17], [49, 0, 6], [35, 9, 23], [12, 17, 11], [3, 8, 16], [14, 27, 15], [20, 47, 2], [8, 32, 28], [27, 12, 17], [46, 3, 8], [25, 9, 27], [12, 38, 19], [2, 6, 13], [9, 39, 12], [17, 44, 2], [8, 24, 31], [39, 15, 41], [37, 2, 6], [16, 42, 27], [15, 26, 43], [5, 8, 32], [10, 21, 12], [20, 48, 7], [6, 24, 9], [21, 12, 38], [37, 5, 8], [16, 31, 40], [15, 38, 34], [5, 8, 25], [9, 23, 12], [20, 1, 7], [6, 24, 18], [39, 12, 20], [34, 2, 8], [13, 28, 39], [12, 41, 46], [5, 8, 25], [10, 40, 15], [20, 48, 2], [8, 25, 31], [30, 12, 26], [49, 3, 6], [16, 18, 27], [12, 26, 11], [0, 6, 13], [4, 22, 12], [26, 45, 3], [6, 24, 42], [30, 15, 17], [1, 3, 6], [13, 18, 27], [12, 20, 46], [2, 8, 25], [9, 21, 15], [26, 37, 2], [6, 25, 29], [40, 12, 26], [47, 0, 8], [16, 14, 30], [12, 20, 19], [7, 6, 13], [36, 21, 15], [26, 34, 0], [6, 24, 28], [39, 12, 20], [49, 0, 8], [32, 9, 30], [15, 38, 49], [7, 8, 35], [33, 39, 15], [17, 49, 2], [6, 16, 10], [40, 15, 41], [49, 5, 8], [25, 4, 21], [15, 26, 45], [3, 6, 35], [29, 22, 12], [26, 49, 2], [6, 25, 28], [27, 15, 41], [34, 2, 8], [25, 14, 40], [15, 17, 46], [3, 8, 32], [29, 30, 15], [41, 48, 0], [8, 16, 33], [40, 12, 20], [43, 0, 6], [24, 9, 30], [15, 20, 34], [5, 8, 13], [14, 30, 12], [17, 34, 5], [6, 25, 4], [30, 12, 26], [37, 2, 6], [32, 4, 27], [15, 20, 45], [5, 6, 32], [42, 22, 12], [26, 19, 2], [6, 25, 14], [39, 12, 26], [43, 7, 6]]



    
    
    
    
    #[[44, 18, 34, 17, 43, 42], [44, 7, 5, 6, 43, 42], [21, 31, 5, 20, 43, 37], [27, 25, 5, 14, 43, 13], [27, 3, 5, 22, 43, 32], [21, 45, 5, 40, 43, 2], [27, 18, 34, 40, 43, 32], [21, 30, 5, 4, 43, 42], [44, 7, 5, 6, 43, 36], [39, 31, 34, 16, 43, 38], [39, 31, 5, 19, 43, 13], [21, 18, 5, 11, 43, 36], [21, 41, 34, 17, 43, 38], [39, 25, 34, 20, 43, 26], [27, 18, 5, 4, 43, 42], [27, 28, 34, 19, 43, 26], [21, 45, 5, 11, 43, 37], [21, 31, 5, 1, 43, 38], [21, 45, 5, 19, 43, 23], [39, 28, 34, 20, 43, 15], [21, 7, 5, 11, 43, 37], [21, 29, 5, 6, 43, 36], [27, 24, 5, 4, 43, 23], [44, 45, 5, 20, 43, 42], [21, 9, 5, 11, 43, 0], [27, 3, 34, 10, 43, 0], [21, 45, 5, 17, 43, 35], [27, 25, 5, 14, 43, 42], [39, 3, 34, 20, 43, 0], [27, 33, 5, 11, 43, 2], [44, 9, 34, 40, 43, 37], [44, 29, 5, 6, 43, 0], [27, 45, 5, 4, 43, 37], [27, 24, 34, 14, 43, 12], [21, 9, 5, 19, 43, 35], [39, 31, 5, 20, 43, 36], [21, 41, 34, 14, 43, 0], [44, 3, 5, 17, 43, 23], [27, 18, 5, 16, 43, 2], [27, 9, 5, 10, 43, 42], [39, 29, 34, 40, 43, 12], [27, 9, 5, 16, 43, 0], [39, 45, 5, 10, 43, 15], [21, 24, 34, 4, 43, 0], [39, 33, 5, 19, 43, 35], [44, 7, 34, 11, 43, 26], [27, 30, 34, 6, 43, 8], [44, 24, 34, 16, 43, 32], [27, 29, 34, 40, 43, 35], [39, 33, 34, 20, 43, 38], [27, 30, 5, 14, 43, 32], [27, 30, 5, 4, 43, 0], [27, 3, 34, 10, 43, 38], [39, 28, 34, 16, 43, 2], [44, 9, 5, 4, 43, 0], [39, 25, 5, 16, 43, 37], [21, 31, 5, 40, 43, 12], [27, 24, 5, 22, 43, 0], [21, 3, 34, 20, 43, 32], [27, 9, 5, 4, 43, 35], [44, 33, 5, 16, 43, 15], [39, 24, 5, 6, 43, 23], [27, 7, 5, 16, 43, 13], [44, 41, 34, 17, 43, 35], [44, 7, 34, 16, 43, 13], [21, 9, 34, 20, 43, 15], [21, 41, 5, 14, 43, 2], [27, 9, 5, 11, 43, 37], [44, 31, 34, 17, 43, 2], [39, 33, 34, 6, 43, 12], [44, 41, 5, 19, 43, 12], [27, 9, 5, 40, 43, 15], [44, 41, 34, 22, 43, 2], [21, 41, 5, 6, 43, 35], [27, 30, 34, 20, 43, 38], [27, 30, 34, 11, 43, 13], [27, 29, 34, 11, 43, 23], [44, 45, 34, 19, 43, 36], [21, 9, 5, 20, 43, 35], [21, 31, 34, 6, 43, 23], [44, 41, 34, 14, 43, 32], [44, 30, 34, 11, 43, 35], [27, 29, 34, 1, 43, 2], [27, 30, 34, 20, 43, 35], [39, 25, 5, 10, 43, 23], [44, 41, 34, 20, 43, 13], [21, 33, 34, 14, 43, 0], [39, 33, 34, 4, 43, 37], [39, 7, 34, 11, 43, 36], [21, 7, 5, 40, 43, 36], [21, 3, 5, 22, 43, 8], [39, 24, 34, 6, 43, 12], [39, 24, 5, 19, 43, 23], [21, 45, 34, 17, 43, 15], [27, 41, 5, 16, 43, 38], [39, 18, 34, 17, 43, 32], [44, 18, 34, 20, 43, 0], [39, 24, 5, 20, 43, 38], [39, 3, 34, 19, 43, 35], [39, 30, 5, 10, 43, 12], [21, 18, 5, 16, 43, 37], [44, 45, 34, 16, 43, 36], [27, 25, 34, 22, 43, 8], [39, 33, 34, 14, 43, 15], [21, 28, 5, 19, 43, 26], [27, 33, 5, 20, 43, 12], [44, 30, 5, 40, 43, 32], [21, 18, 5, 1, 43, 0], [27, 33, 34, 22, 43, 13], [39, 24, 34, 19, 43, 2], [44, 30, 5, 16, 43, 23], [21, 29, 34, 4, 43, 32], [44, 25, 5, 11, 43, 2], [21, 28, 34, 40, 43, 35], [21, 7, 5, 16, 43, 8], [21, 28, 34, 6, 43, 8], [27, 24, 34, 11, 43, 35], [39, 7, 34, 22, 43, 35], [21, 25, 5, 20, 43, 23], [39, 9, 34, 10, 43, 26], [44, 33, 5, 10, 43, 8], [21, 24, 34, 16, 43, 38], [39, 45, 5, 4, 43, 37], [44, 45, 5, 1, 43, 38], [39, 28, 34, 17, 43, 23], [44, 3, 5, 14, 43, 0], [39, 33, 5, 6, 43, 0], [39, 24, 5, 6, 43, 36], [39, 9, 34, 22, 43, 32], [27, 31, 34, 20, 43, 36], [39, 28, 34, 1, 43, 0], [27, 9, 5, 40, 43, 42], [39, 29, 34, 11, 43, 0], [27, 28, 5, 14, 43, 35], [27, 30, 34, 16, 43, 15], [27, 45, 5, 20, 43, 12], [27, 3, 5, 14, 43, 23], [27, 24, 5, 40, 43, 26], [21, 25, 5, 4, 43, 42], [27, 7, 34, 20, 43, 35], [21, 9, 34, 19, 43, 32], [44, 28, 5, 10, 43, 35], [27, 28, 34, 40, 43, 15], [44, 28, 5, 1, 43, 35], [39, 18, 34, 16, 43, 23], [39, 25, 5, 22, 43, 0], [27, 41, 5, 1, 43, 8], [44, 25, 34, 6, 43, 42], [44, 18, 34, 1, 43, 13], [39, 30, 5, 10, 43, 38], [27, 7, 34, 11, 43, 32], [44, 41, 34, 22, 43, 0], [27, 7, 34, 10, 43, 37], [39, 41, 5, 10, 43, 0], [21, 24, 34, 14, 43, 37], [21, 30, 5, 6, 43, 12], [27, 9, 34, 4, 43, 26], [44, 7, 5, 14, 43, 36], [39, 9, 34, 10, 43, 0], [39, 30, 5, 11, 43, 0], [21, 41, 34, 4, 43, 26], [27, 7, 34, 19, 43, 12], [21, 25, 5, 6, 43, 32], [44, 28, 5, 11, 43, 23], [27, 30, 34, 6, 43, 35], [44, 9, 5, 4, 43, 8], [39, 7, 34, 40, 43, 2], [21, 25, 34, 22, 43, 23], [27, 31, 5, 22, 43, 42], [44, 25, 34, 16, 43, 2], [44, 18, 34, 4, 43, 36], [27, 25, 5, 10, 43, 15], [21, 7, 34, 14, 43, 15], [21, 24, 34, 6, 43, 32], [27, 18, 34, 6, 43, 0], [39, 29, 34, 1, 43, 12], [39, 41, 34, 16, 43, 2], [39, 45, 5, 11, 43, 8], [27, 3, 34, 16, 43, 37], [27, 9, 5, 1, 43, 42], [39, 3, 5, 4, 43, 42], [44, 18, 5, 17, 43, 0], [21, 29, 5, 16, 43, 42], [44, 31, 5, 1, 43, 35], [39, 9, 34, 19, 43, 15], [21, 33, 34, 19, 43, 13], [27, 45, 34, 4, 43, 15], [21, 45, 34, 19, 43, 23], [27, 33, 5, 14, 43, 37], [27, 31, 34, 14, 43, 8], [21, 18, 5, 1, 43, 37], [21, 25, 34, 11, 43, 36], [27, 3, 5, 4, 43, 0], [27, 24, 5, 4, 43, 26], [27, 18, 5, 14, 43, 23], [21, 3, 5, 40, 43, 26], [27, 7, 5, 40, 43, 42], [27, 28, 5, 40, 43, 13], [27, 25, 34, 22, 43, 0], [39, 24, 34, 1, 43, 12]]
    #list_l = list(np.arange(0,num_clients))
    #for try_2 in range(0,n_iter):
    #    random.shuffle(list_l)
    #    picked_model.append(list(list_l[0:int(np.ceil(num_clients*top/100))]))
        
    print(picked_model)
        
    loss_f=loss_classifier

    init_lr = lr
    

        
    
    #loss_hist=[[float(loss_dataset(model, training_sets[dl], loss_f).detach()) 
    #    for dl in training_sets]]
    acc_hist, acc_num=accuracy_dataset_per_label(model, test_ld,7)#[[accuracy_dataset(model, testing_sets[dl]) for dl in testing_sets]]
    #server_hist=[[tens_param.detach().numpy() 
    #    for tens_param in list(model.parameters())]]
    models_hist = []
    
    
    for try_1 in range(0,3):


        #model = Model().to(DEVICE)
        
        model_name = 'densenet'
        num_classes = 7
        feature_extract = False
        input_size = 224
        # Initialize the model for this run
        model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
        # Define the device:
        #device = torch.device('cuda:0')
        # Put the model on the device:
        model = model_ft.to(DEVICE)
        
        lr = init_lr

        for i in range(n_iter):
            
            list_select = picked_model[i]
    
            
            clients_params=[]
            clients_models=[]
            clients_losses=[]
            
            K=len(list_select) #number of clients
            n_samples=sum([len(training_sets[db]) for db in list_select])
            weights=([len(training_sets[db])/n_samples for db in list_select])
            print("Clients' weights:",weights)
            
            for k in list_select:
    
                print(k)
                
                local_model=deepcopy(model)
                local_optimizer=optim.SGD(local_model.parameters(),lr=lr)#,weight_decay=0.001)
                
                local_loss=local_learning(local_model,mu,local_optimizer,
                    training_sets[k],epochs,loss_f)
                
                clients_losses.append(local_loss)
                    
                #GET THE PARAMETER TENSORS OF THE MODEL
                list_params=list(local_model.parameters())
                list_params=[tens_param.detach() for tens_param in list_params]
                clients_params.append(list_params)    
                #clients_models.append(deepcopy(local_model))
            
            
            #CREATE THE NEW GLOBAL MODEL
            model = average_models(deepcopy(model), clients_params, 
                weights=weights)
            #models_hist.append(clients_models)
            
            #COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
            #loss_hist+=[[float(loss_dataset(model, training_sets[dl], loss_f).detach()) 
            #    for dl in training_sets]]
            #acc_hist=[[accuracy_dataset(model, testing_sets[dl]) for dl in testing_sets]]
    
            server_loss=0#sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
            server_acc, acc_num=accuracy_dataset_per_label(model, test_ld,7)#sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])
            

            print(f'====> i: {i+1} Server Test Accuracy: {acc_num}')
    
    
            with open('1_X_FLIPS'+str(how_many_parts)+'FLIPS_dir_'+str(int(alpha*10))+'sk_c_dataset'+str(int(alpha*100))+str(top)+'fedavg.csv', 'a') as f:
                # create the csv writer
                writer = csv.writer(f)
    
                # write a row to the csv file
                writer.writerow([str(i),str(server_loss),str(server_acc[0]),str(server_acc[1]),str(server_acc[2]),str(server_acc[3]),str(server_acc[4]),str(server_acc[5]),str(server_acc[6]),str(mu),str(acc_num),str(alpha)])#str(i)+"," + str(server_loss)+","+str(server_acc)+","+str(mu))
                #writer.write("\n")
            
    
            #server_hist.append([tens_param.detach().cpu().numpy() 
            #    for tens_param in list(model.parameters())])
            
            #DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
            lr = lr*decay if i%30==0 else lr
            
    return model, loss_hist, acc_hist

n_iter=400


alphas = [0.3,0.6]

all_parts = [20,15,10,5]


for alpha in alphas:
    for how_many_parts in all_parts:


        #alpha = 0.3

        with open('sc_data_train'+str(int(alpha*10))+'.pickle', 'rb') as handle:
            X_dict_train = pickle.load(handle)


        train_dl = dict()
        test_dl = dict()

        test_dataset = pd.read_csv('all_test_'+str(int(alpha*10))+'_data.csv',header=0)
        normMean = [0.76303697, 0.54564005, 0.57004493]
        normStd = [0.14092775, 0.15261292, 0.16997]
        train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),
                                                transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                                transforms.ToTensor(),
                                                transforms.Normalize(normMean, normStd)])

        testing_set = CustomDataset(test_dataset.drop_duplicates('path'), transform=train_transform)
        test_loader = data.DataLoader(testing_set, batch_size=64, shuffle=True, num_workers=4)

        for key in X_dict_train:
            train_dataset = CustomDataset(X_dict_train[key].drop_duplicates('path'), transform=train_transform)
            train_dataloader = data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
            #test_dataloader = data.DataLoader(test_dataset, batch_size=8, shuffle=False)

            train_dl[key] = train_dataloader 
            #test_dl[key] = test_dataloader

        model_f, loss_hist_FA_iid, acc_hist_FA_iid = FedProx(alpha,model_0, test_loader,
            train_dl, n_iter, test_dl, epochs = 1, 
            lr = 0.001, decay=0.9,mu=0,top=how_many_parts)



'''
for alpha in alphas:
    
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    netdata_idx_map_test = dict()
    netdata_idx_map_train = dict()

    if alpha > 0:
        netdata_idx_map_train = partition_dataset(alpha, y_train, num_clients)
        with open('alpha_dir_'+str(int(alpha*100))+'.pickle', 'rb') as handle:
            netdata_idx_map_train = pickle.load(handle)

        netdata_idx_map_test = partition_dataset(alpha, y_test, num_clients)
    else:
        netdata_idx_map_train,netdata_idx_map_test = partition_alpha_zero(trainset,testset,key_dir='/root/rahul_code/cifar/key_alpha_0.csv')
    
    
    for key in netdata_idx_map_train:
        train_dl, test_dl = get_dataloader('', '', train_bs, test_bs, dataidxs_train=netdata_idx_map_train[key],dataidxs_test=netdata_idx_map_test[key])
    
        train_data_local_dict[key] = train_dl
        test_data_local_dict[key] = test_dl
    
    model_f, loss_hist_FA_iid, acc_hist_FA_iid = FedProx(model_0, 
        train_data_local_dict, n_iter, test_data_local_dict, epochs = 1, 
        lr = 0.001, decay=0.9,mu=0.3,top=how_many_parts)
'''