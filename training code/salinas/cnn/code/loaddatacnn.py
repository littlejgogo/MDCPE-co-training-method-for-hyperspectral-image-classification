import scipy.io as sio
import numpy as np
import tensorflow as tf
import collections
import math
# from PIL import Image
#read cnn data and rnn data

matpath = "/home/asdf/Documents/juyan/data/salinas/cnn/salinas3.mat"
labelpath = "/home/asdf/Documents/juyan/data/salinas/cnn/newsalinas_gt.mat"
readdata = sio.loadmat(matpath)
readlabel = sio.loadmat(labelpath)


datas = readdata.get('train')
label = readlabel.get('label')

image = datas.astype(np.float32)
image = np.transpose(image, (2, 0, 1))
#normalization [x-min]/[max-min]
normdata = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=np.float32)
for dim in range(image.shape[0]):
    normdata[dim, :, :] = (image[dim, :, :] - np.amin(image[dim, :, :])) / \
                          float((np.amax(image[dim, :, :]) - np.amin(image[dim, :, :])))
# label = labels[label]
#change rnn data dimension [w,h,dimendions] to [w*h,channels]

line = label.shape[0]
col = label.shape[1]
# win = 17

datasets = []
labelsets = []

#choose 90% samples to train and 10% samples to valid,
for i in range(line):
    for j in range(col):
        if label[i][j] != 255:
            block = normdata[:, i:i + 17, j:j + 17]
            datasets.append(block)
            labelsets.append(label[i][j])
        else:
            continue
datasets = np.array(datasets)
labelsets = np.array(labelsets)



# np.savez('CNN_salinas.npz', trainimages=traindata, trainlabels=trainlabel, testimages=valdata, testlabels=vallabel)
np.savez('CNN_salinas_datasets.npz', testimages=datasets, testlabels=labelsets)
