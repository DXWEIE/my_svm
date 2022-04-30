import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import numpy as np
from sklearn import svm
import random
import time
import copy
import datetime
import os
import math

random.seed(1)
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

train_dataset = datasets.MNIST(root="../data/", train=True, download=False,
                               transform=transforms.Compose([
                                   transforms.ToTensor(), transforms.Normalize((0.1037,), (0.3081,))]))
test_dataset = datasets.MNIST(root="../data/", train=False, transform=transforms.Compose([
    transforms.ToTensor(),transforms.Normalize((0.1037,), (0.3081,))]))

train_loader = torch.utils.data.DataLoader(train_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset)

train_class_counter = np.zeros(10, dtype=int)
train_class_index = [[] for i in range(10)]
for i in range(train_dataset.data.shape[0]):
    train_class_counter[train_dataset.targets[i]] += 1
    train_class_index[train_dataset.targets[i]].append(i)

test_class_counter = np.zeros(10, dtype=int)
test_class_index = [[] for i in range(10)]

for i in range(test_dataset.data.shape[0]):
    test_class_counter[test_dataset.targets[i]] += 1
    test_class_index[test_dataset.targets[i]].append(i)

#train_ratio = [400, 400, 400, 400, 400, 400, 400, 400, 400, 400]
#test_ratio = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]  # 平均准确率

#train_ratio = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
#test_ratio = [800, 800, 800, 800, 800, 800, 800, 800, 800, 800]  # 平均准确率


train_ratio = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500]
test_ratio = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500]

train_index = [[] for i in range(10)]
for i in range(10):  # 10 classes
    choosing_index = list(range(train_class_counter[i]))
    random.shuffle(choosing_index)
    for j in range(train_ratio[i]):
        train_index[i].append(train_class_index[i][choosing_index[j]])


test_index = [[] for i in range(10)]
for i in range(10):  # 10 classes
    choosing_index = list(range(test_class_counter[i]))
    random.shuffle(choosing_index)
    for j in range(test_ratio[i]):
        test_index[i].append(test_class_index[i][choosing_index[j]])


# train_dataset.data.shape = torch.Size([60000, 28, 28])
new_train = train_dataset.data[train_index[0]]
new_train_targets = train_dataset.targets[train_index[0]]
for i in range(1, 10):
    new_train = torch.cat((new_train, train_dataset.data[train_index[i]]), dim=0)
    new_train_targets = torch.cat((new_train_targets, train_dataset.targets[train_index[i]]), dim=0)

# test_dataset.data.shape = torch.Size([10000, 28, 28])
new_test = test_dataset.data[test_index[0]]
new_test_targets = test_dataset.targets[test_index[0]]
for i in range(1, 10):
    new_test = torch.cat((new_test, test_dataset.data[test_index[i]]), dim=0)
    new_test_targets = torch.cat((new_test_targets, test_dataset.targets[test_index[i]]), dim=0)


train_dataset.data = new_train
train_dataset.targets = new_train_targets

test_dataset.data = new_test
test_dataset.targets = new_test_targets

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=1,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1, shuffle=False)

train_size = sum(train_ratio)
test_size = sum(test_ratio)
train_x = train_dataset.data.numpy().reshape(train_size, 784)
train_y = train_dataset.targets.numpy()
img_shape = 784

test_x = test_dataset.data.numpy().reshape(test_size, 784)
test_y = test_dataset.targets.numpy()



#model = svm.LinearSVC(C=0.5,loss='squared_hinge',fit_intercept=False)
#"""
model = svm.SVC(C=1.0, kernel='poly', degree=3, gamma='auto',
                coef0=0.0, shrinking=True, probability=False, tol=0.001,
                cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                decision_function_shape='ovo', random_state=None)
#"""
model.fit(train_x, train_y)

z = model.predict(test_x)

print('准确率:', np.sum(z == test_y) / z.size)





