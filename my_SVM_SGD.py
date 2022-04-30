import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import numpy as np
import random
import time
import copy
import datetime
import os
from sklearn.preprocessing import StandardScaler
import logging
logger = logging.getLogger("logger")
current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
logger.addHandler(logging.FileHandler('log_'+current_time+'.txt'))
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)
np.random.seed(1)


transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="../../data", train=True, download=False,
                               transform=transforms.Compose([
                                   transforms.ToTensor(), transforms.Normalize((0.1037,), (0.3081,))]))
test_dataset = datasets.MNIST(root="../../data", train=False, transform=transforms.Compose([
    transforms.ToTensor(),transforms.Normalize((0.1037,), (0.3081,))]))


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

#train_ratio = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
train_ratio = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500]
test_ratio = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500]  # 平均准确率

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

w = np.zeros((45,784))  # 45个分类器
b = np.zeros(45)
eta = 0.001 # 0.001
C = 1.0
logger.info('eta = {}'.format(eta))


def index_map(index):
    i = index / 10
    j = index % 10
    return i,j


class best_model_class:
    acc = 0.
    w = None
    b = None

best_model = list()
for i in range(45):
    best_model.append(best_model_class())

T = 31
cls = -1
best_acccc = 0.
for epoch in range(1, T):
    cls = -1
    for i in range(10): # 10个类别
        for j in range(i+1, 10):  # 另外10个类别
            cls += 1
            correct = 0
            datasize = 0
            for batch_id, batch in enumerate(train_loader):
                data, targets = batch
                if targets == i:
                    y = +1
                elif targets == j:
                    y = -1
                else:
                    continue
                x = data.reshape(-1,).numpy()  # 784
                if w[cls].dot(x) + b[cls] >0:
                    pred = +1
                else:
                    pred = -1
                # hinge_loss
                if y * ( w[cls].dot(x) + b[cls]) <= 1:
                    w[cls] = (1-eta) * w[cls] + eta * C * y * x
                    b[cls] = (1-eta) * b[cls] + eta * y
                else:
                    w[cls] = (1-eta) * w[cls]
                    b[cls] = (1-eta) * b[cls]
                datasize += 1
                if pred == y:
                    correct += 1

            acc = float(correct) / float(datasize) * 100

            if acc > best_model[cls].acc:
                best_model[cls].acc = acc
                best_model[cls].w = w[cls]
                best_model[cls].b = b[cls]

            
            logger.info('[epoch {}][class {}{}][Train] acc is {:2f}%'.format(epoch, i,j, acc))


    # test
    correct = 0
    datasize = 0

    for batch_id, batch in enumerate(train_loader):
        vote = np.zeros(10)
        data, targets = batch
        datasize += 1
        x = data.reshape(-1,).numpy()  # 784
        cls = -1
        for i in range(10): # 10个类别
            for j in range(i+1, 10):  # 另外10个类别
                cls += 1
                if np.sign(best_model[cls].w.dot(x) + best_model[cls].b) > 0:
                    vote[i] += 1
                else:
                    vote[j] += 1
        if np.argmax(vote) == targets:
            correct += 1

    acc = float(correct) / float(datasize) * 100
    logger.info('\n[epoch {}][Train] acc is {:2f}%'.format(epoch, acc))
    
    for batch_id, batch in enumerate(test_loader):
        vote = np.zeros(10)
        data, targets = batch
        datasize += 1
        x = data.reshape(-1,).numpy()  # 784
        cls = -1
        for i in range(10): # 10个类别
            for j in range(i+1, 10):  # 另外10个类别
                cls += 1
                if np.sign(best_model[cls].w.dot(x) + best_model[cls].b) > 0:
                    vote[i] += 1
                else:
                    vote[j] += 1
        if np.argmax(vote) == targets:
            correct += 1

    acc = float(correct) / float(datasize) * 100
    if acc>best_acccc:
        best_acccc = acc
    logger.info('[epoch {}][Test] acc is {:2f}%\n'.format(epoch, acc))


logger.info('[Best Test] acc is {:2f}%\n'.format(best_acccc))
    
