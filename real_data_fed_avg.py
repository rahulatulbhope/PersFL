
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


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.utils.data as data
import pandas as pd
from PIL import Image
import math
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

how_many_parts = int(sys.argv[1])
alpha = float(sys.argv[2])
#(X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts) = partition_data('cifar10','/Users/rahulbhope/Documents/rahul_code/cifar','/Users/rahulbhope/Documents/rahul_code/cifar','hetero-dir',100,alpha=alpha)
#train_dl, test_dl = get_dataloader('cifar10', '/Users/rahulbhope/Documents/rahul_code/cifar', 8, 8, dataidxs=net_dataidx_map[1])
how_many_clients = 100
how_many_labels = 5
train_bs = 8
test_bs = 8
#alpha = 0
num_clients = 43

class ECG_dataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.float32)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.x)



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








# build the CNN model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # the first convolution layer, 4 21x1 convolution kernels, output shape (batch_size, 4, 300)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=21, stride=1, padding='same')
        # the first pooling layer, max pooling, pooling size=3 , stride=2, output shape (batch_size, 4, 150)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # the second convolution layer, 16 23x1 convolution kernels, output shape (batch_size, 16, 150)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=23, stride=1, padding='same')
        # the second pooling layer, max pooling, pooling size=3, stride=2, output shape (batch_size, 16, 75)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # the third convolution layer, 32 25x1 convolution kernels, output shape (batch_size, 32, 75)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=25, stride=1, padding='same')
        # the third pooling layer, average pooling, pooling size=3, stride=2, output shape (batch_size, 32, 38)
        self.pool3 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        # the fourth convolution layer, 64 27x1 convolution kernels, output shape (batch_size, 64, 38)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=27, stride=1, padding='same')
        # flatten layer, for the next fully connected layer, output shape (batch_size, 38*64)
        self.flatten = nn.Flatten()
        # fully connected layer, 128 nodes, output shape (batch_size, 128)
        self.fc1 = nn.Linear(64 * 38, 128)
        # Dropout layer, dropout rate = 0.2
        self.dropout = nn.Dropout(0.2)
        # fully connected layer, 5 nodes (number of classes), output shape (batch_size, 5)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        # x.shape = (batch_size, 300)
        # reshape the tensor with shape (batch_size, 300) to (batch_size, 1, 300)
        x = x.reshape(-1, 1, 300)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x





model_0 = Model().to(DEVICE)   

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
        
    return np.nan_to_num(cm_matrix_1), accuracy

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


def FedProx(model, test_ld, training_sets:dict, n_iter:int, testing_sets:dict, mu=0, 
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
    
    picked_model = []#[[34, 28, 42, 18, 43, 38], [22, 26, 42, 18, 49, 38], [48, 28, 42, 18, 17, 38], [20, 19, 42, 18, 45, 38], [25, 19, 42, 18, 43, 38], [32, 28, 42, 18, 49, 38], [20, 28, 42, 18, 43, 38], [3, 26, 42, 18, 39, 38], [34, 28, 42, 18, 33, 38], [24, 29, 42, 18, 49, 38], [41, 21, 42, 18, 17, 38], [23, 26, 42, 18, 40, 38], [8, 29, 42, 18, 45, 38], [10, 19, 42, 18, 15, 38], [27, 26, 42, 18, 43, 38], [47, 26, 42, 18, 39, 38], [8, 29, 42, 18, 43, 38], [6, 26, 42, 18, 49, 38], [22, 26, 42, 18, 39, 38], [3, 19, 42, 18, 15, 38], [8, 26, 42, 18, 49, 38], [27, 21, 42, 18, 43, 38], [14, 21, 42, 18, 39, 38], [22, 26, 42, 18, 49, 38], [12, 29, 42, 18, 33, 38], [37, 19, 42, 18, 17, 38], [30, 19, 42, 18, 49, 38], [13, 21, 42, 18, 49, 38], [32, 28, 42, 18, 49, 38], [37, 29, 42, 18, 43, 38], [2, 28, 42, 18, 33, 38], [24, 28, 42, 18, 15, 38], [1, 26, 42, 18, 33, 38], [30, 21, 42, 18, 45, 38], [6, 29, 42, 18, 49, 38], [23, 26, 42, 18, 43, 38], [46, 29, 42, 18, 45, 38], [12, 19, 42, 18, 15, 38], [41, 29, 42, 18, 45, 38], [36, 26, 42, 18, 39, 38], [25, 19, 42, 18, 40, 38], [11, 28, 42, 18, 49, 38], [24, 21, 42, 18, 40, 38], [44, 19, 42, 18, 45, 38], [44, 21, 42, 18, 16, 38], [27, 21, 42, 18, 17, 38], [1, 28, 42, 18, 17, 38], [34, 26, 42, 18, 16, 38], [13, 21, 42, 18, 16, 38], [2, 26, 42, 18, 33, 38], [44, 29, 42, 18, 33, 38], [35, 21, 42, 18, 33, 38], [2, 26, 42, 18, 16, 38], [24, 26, 42, 18, 43, 38], [7, 26, 42, 18, 49, 38], [6, 28, 42, 18, 15, 38], [10, 29, 42, 18, 17, 38], [6, 26, 42, 18, 45, 38], [7, 26, 42, 18, 45, 38], [9, 26, 42, 18, 45, 38], [24, 26, 42, 18, 16, 38], [4, 29, 42, 18, 39, 38], [25, 26, 42, 18, 16, 38], [2, 29, 42, 18, 33, 38], [12, 21, 42, 18, 43, 38], [27, 21, 42, 18, 15, 38], [30, 19, 42, 18, 17, 38], [27, 29, 42, 18, 15, 38], [46, 21, 42, 18, 45, 38], [14, 21, 42, 18, 16, 38], [27, 19, 42, 18, 43, 38], [11, 21, 42, 18, 33, 38], [8, 26, 42, 18, 15, 38], [1, 26, 42, 18, 49, 38], [5, 26, 42, 18, 16, 38], [41, 19, 42, 18, 43, 38], [24, 29, 42, 18, 45, 38], [47, 19, 42, 18, 45, 38], [34, 28, 42, 18, 39, 38], [13, 19, 42, 18, 33, 38], [48, 28, 42, 18, 45, 38], [9, 29, 42, 18, 17, 38], [20, 26, 42, 18, 49, 38], [12, 21, 42, 18, 17, 38], [24, 29, 42, 18, 17, 38], [24, 29, 42, 18, 16, 38], [44, 28, 42, 18, 39, 38], [4, 19, 42, 18, 17, 38], [23, 26, 42, 18, 49, 38], [9, 19, 42, 18, 16, 38], [10, 28, 42, 18, 43, 38], [41, 21, 42, 18, 49, 38], [46, 28, 42, 18, 49, 38], [34, 19, 42, 18, 43, 38], [0, 29, 42, 18, 40, 38], [36, 19, 42, 18, 43, 38], [4, 19, 42, 18, 16, 38], [23, 19, 42, 18, 39, 38], [23, 28, 42, 18, 43, 38], [25, 29, 42, 18, 15, 38], [1, 26, 42, 18, 17, 38], [2, 28, 42, 18, 16, 38], [12, 19, 42, 18, 39, 38], [22, 19, 42, 18, 16, 38], [37, 26, 42, 18, 49, 38], [7, 19, 42, 18, 45, 38], [14, 26, 42, 18, 15, 38], [10, 21, 42, 18, 49, 38], [10, 21, 42, 18, 43, 38], [25, 21, 42, 18, 16, 38], [36, 19, 42, 18, 39, 38], [44, 19, 42, 18, 17, 38], [25, 21, 42, 18, 15, 38], [11, 29, 42, 18, 40, 38], [14, 28, 42, 18, 43, 38], [24, 26, 42, 18, 49, 38], [44, 21, 42, 18, 43, 38], [8, 21, 42, 18, 40, 38], [35, 21, 42, 18, 40, 38], [3, 26, 42, 18, 16, 38], [6, 28, 42, 18, 40, 38], [3, 26, 42, 18, 49, 38], [5, 29, 42, 18, 39, 38], [37, 29, 42, 18, 17, 38], [0, 29, 42, 18, 17, 38], [31, 28, 42, 18, 40, 38], [14, 26, 42, 18, 39, 38], [4, 29, 42, 18, 45, 38], [3, 29, 42, 18, 49, 38], [25, 21, 42, 18, 40, 38], [0, 26, 42, 18, 40, 38], [25, 21, 42, 18, 17, 38], [32, 26, 42, 18, 45, 38], [48, 19, 42, 18, 45, 38], [11, 29, 42, 18, 15, 38], [22, 28, 42, 18, 43, 38], [2, 26, 42, 18, 33, 38], [37, 28, 42, 18, 45, 38], [2, 26, 42, 18, 33, 38], [23, 21, 42, 18, 39, 38], [5, 19, 42, 18, 33, 38], [35, 21, 42, 18, 17, 38], [34, 28, 42, 18, 43, 38], [4, 28, 42, 18, 43, 38], [22, 28, 42, 18, 16, 38], [22, 26, 42, 18, 15, 38], [34, 28, 42, 18, 39, 38], [27, 26, 42, 18, 40, 38], [5, 19, 42, 18, 16, 38], [22, 29, 42, 18, 40, 38], [44, 21, 42, 18, 43, 38], [2, 21, 42, 18, 15, 38], [32, 21, 42, 18, 39, 38], [10, 19, 42, 18, 45, 38], [20, 21, 42, 18, 17, 38], [22, 26, 42, 18, 16, 38], [47, 21, 42, 18, 39, 38], [12, 21, 42, 18, 49, 38], [5, 21, 42, 18, 45, 38], [7, 28, 42, 18, 40, 38], [9, 21, 42, 18, 40, 38], [27, 21, 42, 18, 17, 38], [48, 26, 42, 18, 45, 38], [31, 26, 42, 18, 45, 38], [25, 19, 42, 18, 43, 38], [44, 19, 42, 18, 33, 38], [0, 29, 42, 18, 39, 38], [20, 29, 42, 18, 15, 38], [32, 21, 42, 18, 39, 38], [30, 29, 42, 18, 17, 38], [20, 19, 42, 18, 39, 38], [1, 29, 42, 18, 33, 38], [36, 26, 42, 18, 49, 38], [47, 29, 42, 18, 15, 38], [7, 28, 42, 18, 43, 38], [3, 21, 42, 18, 16, 38], [12, 26, 42, 18, 43, 38], [44, 28, 42, 18, 16, 38], [4, 28, 42, 18, 17, 38], [27, 28, 42, 18, 45, 38], [24, 19, 42, 18, 15, 38], [37, 19, 42, 18, 45, 38], [41, 21, 42, 18, 33, 38], [32, 19, 42, 18, 39, 38], [1, 21, 42, 18, 33, 38], [3, 21, 42, 18, 49, 38], [37, 28, 42, 18, 49, 38], [32, 26, 42, 18, 45, 38], [6, 28, 42, 18, 43, 38], [5, 26, 42, 18, 49, 38], [2, 28, 42, 18, 45, 38], [0, 19, 42, 18, 33, 38], [32, 19, 42, 18, 45, 38], [3, 29, 42, 18, 43, 38], [10, 21, 42, 18, 49, 38], [36, 29, 42, 18, 16, 38], [24, 19, 42, 18, 33, 38], [24, 28, 42, 18, 16, 38], [35, 19, 42, 18, 49, 38], [3, 21, 42, 18, 17, 38]]
    
    #[[44, 18, 34, 17, 43, 42], [44, 7, 5, 6, 43, 42], [21, 31, 5, 20, 43, 37], [27, 25, 5, 14, 43, 13], [27, 3, 5, 22, 43, 32], [21, 45, 5, 40, 43, 2], [27, 18, 34, 40, 43, 32], [21, 30, 5, 4, 43, 42], [44, 7, 5, 6, 43, 36], [39, 31, 34, 16, 43, 38], [39, 31, 5, 19, 43, 13], [21, 18, 5, 11, 43, 36], [21, 41, 34, 17, 43, 38], [39, 25, 34, 20, 43, 26], [27, 18, 5, 4, 43, 42], [27, 28, 34, 19, 43, 26], [21, 45, 5, 11, 43, 37], [21, 31, 5, 1, 43, 38], [21, 45, 5, 19, 43, 23], [39, 28, 34, 20, 43, 15], [21, 7, 5, 11, 43, 37], [21, 29, 5, 6, 43, 36], [27, 24, 5, 4, 43, 23], [44, 45, 5, 20, 43, 42], [21, 9, 5, 11, 43, 0], [27, 3, 34, 10, 43, 0], [21, 45, 5, 17, 43, 35], [27, 25, 5, 14, 43, 42], [39, 3, 34, 20, 43, 0], [27, 33, 5, 11, 43, 2], [44, 9, 34, 40, 43, 37], [44, 29, 5, 6, 43, 0], [27, 45, 5, 4, 43, 37], [27, 24, 34, 14, 43, 12], [21, 9, 5, 19, 43, 35], [39, 31, 5, 20, 43, 36], [21, 41, 34, 14, 43, 0], [44, 3, 5, 17, 43, 23], [27, 18, 5, 16, 43, 2], [27, 9, 5, 10, 43, 42], [39, 29, 34, 40, 43, 12], [27, 9, 5, 16, 43, 0], [39, 45, 5, 10, 43, 15], [21, 24, 34, 4, 43, 0], [39, 33, 5, 19, 43, 35], [44, 7, 34, 11, 43, 26], [27, 30, 34, 6, 43, 8], [44, 24, 34, 16, 43, 32], [27, 29, 34, 40, 43, 35], [39, 33, 34, 20, 43, 38], [27, 30, 5, 14, 43, 32], [27, 30, 5, 4, 43, 0], [27, 3, 34, 10, 43, 38], [39, 28, 34, 16, 43, 2], [44, 9, 5, 4, 43, 0], [39, 25, 5, 16, 43, 37], [21, 31, 5, 40, 43, 12], [27, 24, 5, 22, 43, 0], [21, 3, 34, 20, 43, 32], [27, 9, 5, 4, 43, 35], [44, 33, 5, 16, 43, 15], [39, 24, 5, 6, 43, 23], [27, 7, 5, 16, 43, 13], [44, 41, 34, 17, 43, 35], [44, 7, 34, 16, 43, 13], [21, 9, 34, 20, 43, 15], [21, 41, 5, 14, 43, 2], [27, 9, 5, 11, 43, 37], [44, 31, 34, 17, 43, 2], [39, 33, 34, 6, 43, 12], [44, 41, 5, 19, 43, 12], [27, 9, 5, 40, 43, 15], [44, 41, 34, 22, 43, 2], [21, 41, 5, 6, 43, 35], [27, 30, 34, 20, 43, 38], [27, 30, 34, 11, 43, 13], [27, 29, 34, 11, 43, 23], [44, 45, 34, 19, 43, 36], [21, 9, 5, 20, 43, 35], [21, 31, 34, 6, 43, 23], [44, 41, 34, 14, 43, 32], [44, 30, 34, 11, 43, 35], [27, 29, 34, 1, 43, 2], [27, 30, 34, 20, 43, 35], [39, 25, 5, 10, 43, 23], [44, 41, 34, 20, 43, 13], [21, 33, 34, 14, 43, 0], [39, 33, 34, 4, 43, 37], [39, 7, 34, 11, 43, 36], [21, 7, 5, 40, 43, 36], [21, 3, 5, 22, 43, 8], [39, 24, 34, 6, 43, 12], [39, 24, 5, 19, 43, 23], [21, 45, 34, 17, 43, 15], [27, 41, 5, 16, 43, 38], [39, 18, 34, 17, 43, 32], [44, 18, 34, 20, 43, 0], [39, 24, 5, 20, 43, 38], [39, 3, 34, 19, 43, 35], [39, 30, 5, 10, 43, 12], [21, 18, 5, 16, 43, 37], [44, 45, 34, 16, 43, 36], [27, 25, 34, 22, 43, 8], [39, 33, 34, 14, 43, 15], [21, 28, 5, 19, 43, 26], [27, 33, 5, 20, 43, 12], [44, 30, 5, 40, 43, 32], [21, 18, 5, 1, 43, 0], [27, 33, 34, 22, 43, 13], [39, 24, 34, 19, 43, 2], [44, 30, 5, 16, 43, 23], [21, 29, 34, 4, 43, 32], [44, 25, 5, 11, 43, 2], [21, 28, 34, 40, 43, 35], [21, 7, 5, 16, 43, 8], [21, 28, 34, 6, 43, 8], [27, 24, 34, 11, 43, 35], [39, 7, 34, 22, 43, 35], [21, 25, 5, 20, 43, 23], [39, 9, 34, 10, 43, 26], [44, 33, 5, 10, 43, 8], [21, 24, 34, 16, 43, 38], [39, 45, 5, 4, 43, 37], [44, 45, 5, 1, 43, 38], [39, 28, 34, 17, 43, 23], [44, 3, 5, 14, 43, 0], [39, 33, 5, 6, 43, 0], [39, 24, 5, 6, 43, 36], [39, 9, 34, 22, 43, 32], [27, 31, 34, 20, 43, 36], [39, 28, 34, 1, 43, 0], [27, 9, 5, 40, 43, 42], [39, 29, 34, 11, 43, 0], [27, 28, 5, 14, 43, 35], [27, 30, 34, 16, 43, 15], [27, 45, 5, 20, 43, 12], [27, 3, 5, 14, 43, 23], [27, 24, 5, 40, 43, 26], [21, 25, 5, 4, 43, 42], [27, 7, 34, 20, 43, 35], [21, 9, 34, 19, 43, 32], [44, 28, 5, 10, 43, 35], [27, 28, 34, 40, 43, 15], [44, 28, 5, 1, 43, 35], [39, 18, 34, 16, 43, 23], [39, 25, 5, 22, 43, 0], [27, 41, 5, 1, 43, 8], [44, 25, 34, 6, 43, 42], [44, 18, 34, 1, 43, 13], [39, 30, 5, 10, 43, 38], [27, 7, 34, 11, 43, 32], [44, 41, 34, 22, 43, 0], [27, 7, 34, 10, 43, 37], [39, 41, 5, 10, 43, 0], [21, 24, 34, 14, 43, 37], [21, 30, 5, 6, 43, 12], [27, 9, 34, 4, 43, 26], [44, 7, 5, 14, 43, 36], [39, 9, 34, 10, 43, 0], [39, 30, 5, 11, 43, 0], [21, 41, 34, 4, 43, 26], [27, 7, 34, 19, 43, 12], [21, 25, 5, 6, 43, 32], [44, 28, 5, 11, 43, 23], [27, 30, 34, 6, 43, 35], [44, 9, 5, 4, 43, 8], [39, 7, 34, 40, 43, 2], [21, 25, 34, 22, 43, 23], [27, 31, 5, 22, 43, 42], [44, 25, 34, 16, 43, 2], [44, 18, 34, 4, 43, 36], [27, 25, 5, 10, 43, 15], [21, 7, 34, 14, 43, 15], [21, 24, 34, 6, 43, 32], [27, 18, 34, 6, 43, 0], [39, 29, 34, 1, 43, 12], [39, 41, 34, 16, 43, 2], [39, 45, 5, 11, 43, 8], [27, 3, 34, 16, 43, 37], [27, 9, 5, 1, 43, 42], [39, 3, 5, 4, 43, 42], [44, 18, 5, 17, 43, 0], [21, 29, 5, 16, 43, 42], [44, 31, 5, 1, 43, 35], [39, 9, 34, 19, 43, 15], [21, 33, 34, 19, 43, 13], [27, 45, 34, 4, 43, 15], [21, 45, 34, 19, 43, 23], [27, 33, 5, 14, 43, 37], [27, 31, 34, 14, 43, 8], [21, 18, 5, 1, 43, 37], [21, 25, 34, 11, 43, 36], [27, 3, 5, 4, 43, 0], [27, 24, 5, 4, 43, 26], [27, 18, 5, 14, 43, 23], [21, 3, 5, 40, 43, 26], [27, 7, 5, 40, 43, 42], [27, 28, 5, 40, 43, 13], [27, 25, 34, 22, 43, 0], [39, 24, 34, 1, 43, 12]]
    list_l = list(np.arange(0,num_clients))
    for try_2 in range(0,n_iter):
        random.shuffle(list_l)
        picked_model.append(list(list_l[0:int(math.ceil(num_clients*top/100))]))
        
    print(picked_model)
        
    loss_f=loss_classifier

    init_lr = lr
    

        
    
    #loss_hist=[[float(loss_dataset(model, training_sets[dl], loss_f).detach()) 
    #    for dl in training_sets]]
    #acc_hist, acc_num=accuracy_dataset_per_label(model, test_ld,5)#[[accuracy_dataset(model, testing_sets[dl]) for dl in testing_sets]]
    #server_hist=[[tens_param.detach().numpy() 
    #    for tens_param in list(model.parameters())]]
    models_hist = []
    
    
    for try_1 in range(0,3):

        model = Model().to(DEVICE)
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
            server_acc, acc_num=accuracy_dataset_per_label(model, test_ld,5)#sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])
            

            print(f'====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}')
    
    
            with open('Real_data_'+str(how_many_parts)+'_dir_'+str(int(alpha*10))+'_redo_non_clus_poison_arrythmia_lenet'+str(int(alpha*100))+str(top)+'fedavg.csv', 'a') as f:
                # create the csv writer
                writer = csv.writer(f)
    
                # write a row to the csv file
                writer.writerow([str(i),str(server_loss),str(server_acc[0]),str(server_acc[1]),str(server_acc[2]),str(server_acc[3]),str(server_acc[4]),str(mu),str(acc_num)])#str(i)+"," + str(server_loss)+","+str(server_acc)+","+str(mu))
                #writer.write("\n")
            
    
            #server_hist.append([tens_param.detach().cpu().numpy() 
            #    for tens_param in list(model.parameters())])
            
            #DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
            lr = lr*decay if i%10==0 else lr
            
    return model, loss_hist, acc_hist

n_iter=200


#alphas = [0.3]

#alpha = 0.3

with open('mit_data_train_43.pickle', 'rb') as handle:
    X_dict_train = pickle.load(handle)

with open('mit_data_labels_train_43.pickle', 'rb') as handle:
    Y_dict_train = pickle.load(handle)


with open('full_test_mit_data_test.pickle', 'rb') as handle:
    X_dict_test= pickle.load(handle)

with open('full_test_mit_data_labels_test.pickle', 'rb') as handle:
    Y_dict_test = pickle.load(handle)



train_dl = dict()
#test_dl = dict()

test_dataset = ECG_dataset(X_dict_test, Y_dict_test)
test_dataloader= data.DataLoader(test_dataset, batch_size=512, shuffle=True)


for key in X_dict_train:
    train_dataset = ECG_dataset(X_dict_train[key], Y_dict_train[key])
    train_dataloader = data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    #test_dataset = ECG_dataset(X_dict_test[key], Y_dict_test[key])
    #test_dataloader= data.DataLoader(test_dataset, batch_size=8, shuffle=True)

    train_dl[key] = train_dataloader 
    #test_dl[key] = test_dataloader

model_f, loss_hist_FA_iid, acc_hist_FA_iid = FedProx(model_0, test_dataloader,
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