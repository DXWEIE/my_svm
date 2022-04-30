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
import logging
logger = logging.getLogger("logger")
current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
logger.addHandler(logging.FileHandler('log_'+current_time+'.txt'))
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

def linear_kernel(x, y):
    return np.dot(x, y)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=0.3081):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

random.seed(1)
torch.manual_seed(1)
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

#train_ratio = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
#test_ratio = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]  # 平均准确率
#train_ratio = [300, 300, 300, 0, 0, 0, 0, 0, 0, 0]
#test_ratio = [200, 200, 200, 0, 0, 0, 0, 0, 0, 0]  # 平均准确率


train_ratio = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500]
test_ratio = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500]  # 平均准确率


#train_ratio = [200, 200, 200, 200, 0, 0, 0, 0, 0, 0]
#test_ratio = [100, 100, 100, 100, 0, 0, 0, 0, 0, 0]  # 平均准确率


#train_ratio = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
#test_ratio = [800, 800, 800, 800, 800, 800, 800, 800, 800, 800]  # 平均准确率

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

TRAINSIZE = sum(train_ratio)
n_features = 784
x_train_data = train_loader.dataset.data.reshape(TRAINSIZE,n_features).float().numpy()
y_train_data = train_loader.dataset.targets.reshape(-1,).float().numpy()

TESTSIZE = sum(test_ratio)
x_test_data = test_loader.dataset.data.reshape(TESTSIZE,n_features).float().numpy()
y_test_data = test_loader.dataset.targets.reshape(-1,).float().numpy()

classes_N = 10  # 类别个数

data_idx = list()
cls = -1
for m in range(classes_N):  # 10个类别
    for n in range(m+1, classes_N):
        cls += 1
        data_idx.append(np.where((y_train_data==m)|(y_train_data==n)))

gamma = 1.0 / n_features
#y_train[y_train==0] = -1
#y_train[y_train>0] = +1


class best_model_class:
    acc = 0.
    alpha = None
    b = None
    E = None
    gx = None


b = np.zeros( int(classes_N * (classes_N-1)/2) )
lr = 0.01
C = 1.0

best_model = list()
Gram = list()
alpha = list()
gx = list()

E = list()
cls = -1
for m in range(classes_N):
    for n in range(m+1, classes_N):
        cls += 1
        
        x_train = x_train_data[data_idx[cls]]
        y_train = np.zeros_like(y_train_data[data_idx[cls]])
        y_train[np.where(y_train_data[data_idx[cls]] ==m)] = 1
        y_train[np.where(y_train_data[data_idx[cls]] ==n)] = -1
        TRAINSIZE = len(y_train)
        temp = np.zeros((TRAINSIZE,TRAINSIZE))
        for i in range(TRAINSIZE):
            for j in range(TRAINSIZE):
                temp[i][j] = linear_kernel(x_train[i], x_train[j])  #x_train[i].dot(x_train[j])

        Gram.append(temp)
        np.save('./gram/gram_'+str(cls),temp)
        print('done[',cls,']')


        temp4 = np.zeros(TRAINSIZE)
        alpha.append(temp4)

        temp2 = np.zeros(TRAINSIZE)
        temp3 = np.zeros(TRAINSIZE)
        for i in range(TRAINSIZE):
            temp2[i] = 0
            for j in range(TRAINSIZE):
                temp2[i] += alpha[cls][j] * y_train[j] * Gram[cls][j][i] + b[cls]

            temp3[i] = temp2[i] - y_train[i]

        gx.append(temp2)
        E.append(temp3)
        best_model.append(best_model_class())

logger.info('done')
T = 51  # 迭代数
for epoch in range(1, T):
    start_time = time.time()
    cls = -1
    for m in range(classes_N):
        for n in range(m+1, classes_N):
            cls += 1
            x_train = x_train_data[data_idx[cls]]
            y_train = np.zeros_like(y_train_data[data_idx[cls]])
            y_train[np.where(y_train_data[data_idx[cls]] ==m)] = 1
            y_train[np.where(y_train_data[data_idx[cls]] ==n)] = -1
            TRAINSIZE = len(y_train)
        
            con1 = alpha[cls] > 0
            con2 = alpha[cls] < C

            err1 = y_train * gx[cls] - 1
            err2 = err1.copy()
            err3 = err1.copy()

            err1[(con1 & (err1 <= 0)) | (~con1 & (err1 > 0))] = 0
            err2[((~con1 | ~con2) & (err2 != 0)) | ((con1 & con2) & (err2 == 0))] = 0
            err3[(con2 & (err3 >= 0)) | (~con2 & (err3 < 0))] = 0

            err = err1 ** 2 + err2 ** 2 + err3 ** 2
            a1_idx = np.argmax(err)

            if E[cls][a1_idx] >= 0:
                a2_idx = np.argmin(E[cls])
            else:
                a2_idx = np.argmax(E[cls])
            if a2_idx== a1_idx:
                a2_idx = np.random.randint(TRAINSIZE)
                while a2_idx == a1_idx:
                    a2_idx = np.random.randint(TRAINSIZE)

            a1 = alpha[cls][a1_idx]
            a2 = alpha[cls][a2_idx]

            if y_train[a1_idx] != y_train[a2_idx]:   
                L = max(0, a2 - a1)
                H = min(C, C + a2 - a1)
            else:
                L = max(0, a2 + a1 - C)
                H = min(C, a2 + a1)
            eta = Gram[cls][a1_idx][a1_idx] + Gram[cls][a2_idx][a2_idx] - 2 * Gram[cls][a1_idx][a2_idx]
            a2_unc = a2 + y_train[a2_idx] * (E[cls][a1_idx] - E[cls][a2_idx]) / eta

            # 剪辑
            if a2_unc > H:
                a2_new = H
            elif L <= a2_unc <= H:
                a2_new = a2_unc
            elif a2_unc< L:
                a2_new = L

            a1_new = a1 + y_train[a1_idx] * y_train[a2_idx] * (a2 - a2_new)

            alpha[cls][a1_idx] = a1_new
            alpha[cls][a2_idx] = a2_new


            if a2_new < 1e-7:
                a2_new = 0.0
            elif a2_new > (C - 1e-7):
                a2_new = C

            b1 = -E[cls][a1_idx] - y_train[a1_idx] * Gram[cls][a1_idx][a1_idx] * (a1_new - a1) \
                            - y_train[a2_idx] * Gram[cls][a2_idx][a1_idx] * (a2_new - a2) + b[cls]

            b2 = -E[cls][a2_idx] - y_train[a1_idx] * Gram[cls][a1_idx][a2_idx] * (a1_new - a1) \
                            - y_train[a2_idx] * Gram[cls][a2_idx][a2_idx] * (a2_new - a2) + b[cls]

            if 0 < a1_new < C:
                b[cls] = b1
            elif  0 < a2_new < C:
                b[cls] = b2
            else:
                b[cls] = (b1 + b2) * 0.5

            for i in range(TRAINSIZE):
                gx[cls][i] = 0
                for j in range(TRAINSIZE):
                    gx[cls][i] += alpha[cls][j] * y_train[j] * Gram[cls][j][i]

                E[cls][i] = gx[cls][i] - y_train[i]

            try:
                star = np.where(alpha[cls]>0)[0][0]

                sum_a = 0.
                for j in range(TRAINSIZE):
                    sum_a += alpha[cls][j] * y_train[j] * Gram[cls][j][star]

                b_new = y_train[star] - sum_a

                datasize = 0
                correct = 0
                for i in range(TRAINSIZE):
                    x = x_train[i]
                    y = y_train[i]


                    sum_a = 0.
                    for j in range(TRAINSIZE):
                        sum_a += alpha[cls][j] * y_train[j] * linear_kernel(x_train[j], x)  #x_train[j].dot(x)
                    
                    if sum_a + b_new >0:
                        pred = 1
                    else:
                        pred = -1

                    if pred == y:
                        correct += 1
                    datasize += 1

                acc = float(correct) / float(TRAINSIZE) * 100

                if acc > best_model[cls].acc:
                    best_model[cls].acc = copy.deepcopy(acc)
                    best_model[cls].alpha = copy.deepcopy(alpha[cls])
                    best_model[cls].b = copy.deepcopy(b[cls])
                    best_model[cls].gx = copy.deepcopy(gx[cls])
                    best_model[cls].E = copy.deepcopy(E[cls])
                
                logger.info('[epoch{}][class{}{}]Train acc is {:.2f}% = {:d}/{:d}'.format(epoch,m,n, acc,correct,datasize))
            except:
                logger.info('[epoch{}][class{}{}]not solution'.format(epoch, m,n,))
    end_time = time.time()
    logger.info('train cost {:.2f}s\n'.format(end_time - start_time))
    
    if epoch % 10 == 0:
        # train dataset
        start_time = time.time()
        correct = 0
        datasize = 0
        for idx in range(sum(train_ratio)):
            vote = np.zeros(10)
            x = x_train_data[idx].reshape(-1,)
            y = y_train_data[idx]
            datasize += 1
            cls = -1
            for m in range(classes_N): # 10个类别
                for n in range(m+1, classes_N):  # 另外10个类别
                    cls += 1
                    x_train = x_train_data[data_idx[cls]]
                    y_train = np.zeros_like(y_train_data[data_idx[cls]])
                    y_train[np.where(y_train_data[data_idx[cls]] == m)] = 1
                    y_train[np.where(y_train_data[data_idx[cls]] == n)] = -1
                    TRAINSIZE = len(y_train)


                    star = np.where(alpha[cls]>0)[0][0]

                    sum_a = 0.
                    for j in range(TRAINSIZE):
                        sum_a += best_model[cls].alpha[j] * y_train[j] * Gram[cls][j][star]

                    b_new = y_train[star] - sum_a

                    sum_a = 0.
                    for j in range(TRAINSIZE):
                        sum_a += best_model[cls].alpha[j] * y_train[j] * linear_kernel(x_train[j], x)  #* x_train[j].dot(x)
                        
                    if sum_a + b_new > 0:
                        pred = 0
                        vote[m] += 1
                    else:
                        pred = 1
                        vote[n] += 1

            if np.argmax(vote) == y:
                correct += 1

        acc = float(correct) / float(datasize) * 100
        logger.info('\n[epoch {}][Train] acc is {:2f}%'.format(epoch, acc))

        end_time = time.time()
        logger.info('train dataset valid cost {:.2f}s\n'.format(end_time - start_time))
        #"""
        # test
        start_time = time.time()
        correct = 0
        datasize = 0
        for idx in range(sum(test_ratio)):
            vote = np.zeros(10)
            x = x_test_data[idx].reshape(-1, )
            y = y_test_data[idx]
            datasize += 1
            cls = -1
            for m in range(classes_N):  # 10个类别
                for n in range(m + 1, classes_N):  # 另外10个类别
                    cls += 1
                    x_train = x_train_data[data_idx[cls]]
                    y_train = np.zeros_like(y_train_data[data_idx[cls]])
                    y_train[np.where(y_train_data[data_idx[cls]] == m)] = 1
                    y_train[np.where(y_train_data[data_idx[cls]] == n)] = -1
                    TRAINSIZE = len(y_train)

                    star = np.where(best_model[cls].alpha > 0)[0][0]

                    sum_a = 0.
                    for j in range(TRAINSIZE):
                        sum_a += best_model[cls].alpha[j] * y_train[j] * Gram[cls][j][star]

                    b_new = y_train[star] - sum_a

                    sum_a = 0.
                    for j in range(TRAINSIZE):
                        sum_a += best_model[cls].alpha[j] * y_train[j] * linear_kernel(x_train[j], x)  #* x_train[j].dot(x)

                    if sum_a + b_new > 0:
                        pred = 0
                        vote[m] += 1
                    else:
                        pred = 1
                        vote[n] += 1

            if np.argmax(vote) == y:
                correct += 1

        acc = float(correct) / float(datasize) * 100
        logger.info('[epoch {}][Test] acc is {:2f}%'.format(epoch, acc))
        end_time = time.time()
        logger.info('test dataset valid cost {:.2f}s\n\n'.format(end_time - start_time))

    #"""
