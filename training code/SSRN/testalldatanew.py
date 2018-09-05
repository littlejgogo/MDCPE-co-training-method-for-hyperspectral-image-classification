from keras.models import load_model
import numpy as np
import scipy.io as sio
import time
from keras.utils.np_utils import to_categorical
from sklearn import metrics, preprocessing
from Utils import zeroPadding, averageAccuracy, modelStatsRecord
from tqdm import tqdm
import cv2
import scipy.misc
import keras.backend.tensorflow_backend as KTF
import os
import tensorflow as tf
from libtiff import TIFF
import collections
predicted = []
CATEGORY = 9

def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch


PATCH_LENGTH = 11
INPUT_DIMENSION_CONV = 103
INPUT_DIMENSION = 103

all_indices = np.load("/home/asdf/Documents/juyan/SSRN-master/juyan/pudata/all_index.npy")
all_data = np.zeros((len(all_indices), 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))



uPavia = sio.loadmat('/home/asdf/Documents/juyan/SSRN-master/datasets/UP/PaviaU.mat')
data_IN = uPavia['paviaU']

data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))
data = preprocessing.scale(data) #standardlize mean=0,std = 1
data_ = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])
whole_data = data_
padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)

all_assign = indexToAssignment(all_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
for i in range(len(all_assign)):
    all_data[i] = selectNeighboringPatch(padded_data, all_assign[i][0], all_assign[i][1], PATCH_LENGTH)

x_all = all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], INPUT_DIMENSION_CONV)

def hsi_test_data(batch_size, patch_size, index_iter):
    hsi_batch_patch = np.zeros((batch_size, patch_size, patch_size, INPUT_DIMENSION), dtype=np.uint8)
    for i in range(batch_size):
        hsi_batch_patch[i] = padded_data[index_iter:index_iter+23, i:i+23, :]
    return hsi_batch_patch

model = load_model('/home/asdf/Documents/juyan/SSRN-master/zw/model/pavia_u/UP_best_RES_3D_SS4_5_1.hdf5')
tic7 = time.clock()
pre_test_pro = np.array([])
pre_test_pro.shape = 0, 9
for index_iter in tqdm(range(whole_data.shape[0])):
    x_test = hsi_test_data(batch_size=340, patch_size=23, index_iter=index_iter)
    pre_test = model.predict(
            x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1))
    pre_test = pre_test.reshape((pre_test.shape[0], 9))
    pre_test_pro = np.concatenate((pre_test_pro, pre_test), 0)
toc7 = time.clock()
test_time = toc7-tic7
# np.save('/home/asdf/Documents/juyan/SSRN-master/zw/result/pavia_u/HSI_0321_labeled_pro.npy', pre_test_pro)
label_mat = np.zeros((610, 340), dtype=np.uint8)
pre_label = pre_test_pro.argmax(axis=1)
index = 0
for ir in tqdm(range(whole_data.shape[0])):
    for ic in range(whole_data.shape[1]):
        label_mat[ir][ic] = pre_label[index]
        index += 1
scipy.misc.imsave('/home/asdf/Documents/juyan/SSRN-master/juyan/pu_0704_.tif', label_mat)


