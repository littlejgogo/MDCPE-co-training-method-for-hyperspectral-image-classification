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
ITER = 1
CATEGORY = 16
# def get_session(gpu_fraction=0.3):
#     """
#     This function is to allocate GPU memory a specific fraction
#     Assume that you have 6GB of GPU memory and want to allocate ~2GB
#     """
#
#     num_threads = os.environ.get('OMP_NUM_THREADS')
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#
#     if num_threads:
#         return tf.Session(config=tf.ConfigProto(
#             gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
#     else:
#         return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#
#
# KTF.set_session(get_session(0.2))  # using 60% of total GPU Memory
# os.system("nvidia-smi")  # Execute the command (a string) in a subshell
# raw_input("Press Enter to continue...")

PATCH_LENGTH = 11
INPUT_DIMENSION_CONV = 204
INPUT_DIMENSION = 204

# uPavia = sio.loadmat('/home/asdf/Documents/juyan/paper/salinas/cnn/data/Salinas_corrected.mat')
# # gt_uPavia = sio.loadmat('/home/asdf/Documents/juyan/paper/paviac/cnn/data/Pavia_gt.mat')
# data_IN = uPavia['salinas_corrected']
test_indices = np.load("/home/asdf/Documents/juyan/paper/salinas/cnn/data/test_index.npy")
uPavia = sio.loadmat('/home/asdf/Documents/juyan/paper/salinas/cnn/data/Salinas_corrected.mat')
gt_uPavia = sio.loadmat('/home/asdf/Documents/juyan/paper/salinas/cnn/data/Salinas_gt.mat')
data_IN = uPavia['salinas_corrected']
gt_IN = gt_uPavia['salinas_gt']

data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))
gt = gt_IN.reshape(np.prod(gt_IN.shape[:2]),)
data = preprocessing.scale(data) #standardlize mean=0,std = 1
data_ = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])
whole_data = data_
padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)

def hsi_test_data(batch_size, patch_size, index_iter):
    hsi_batch_patch = np.zeros((batch_size, patch_size, patch_size, INPUT_DIMENSION), dtype=np.float32)
    for i in range(batch_size):
        hsi_batch_patch[i] = padded_data[index_iter:index_iter+23, i:i+23, :]
    return hsi_batch_patch

model = load_model('/home/asdf/Documents/juyan/SSRN-master/juyan/model/SA_1.hdf5')
pre_test_pro = np.array([])
for index_iter in tqdm(range(whole_data.shape[0])):
    x_test = hsi_test_data(batch_size=217, patch_size=23, index_iter=index_iter)
    pre_test = model.predict(
            x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1)).argmax(axis=1)
    # pre_test = pre_test.reshape((pre_test.shape[0], 16))
    pre_test_pro = np.concatenate((pre_test_pro, pre_test), 0)
#
KAPPA_RES_SS4 = []
OA_RES_SS4 = []
AA_RES_SS4 = []
TRAINING_TIME_RES_SS4 = []
TESTING_TIME_RES_SS4 = []
ELEMENT_ACC_RES_SS4 = np.zeros((ITER, CATEGORY))
# pre_test_pro = np.load("/home/asdf/Documents/juyan/SSRN-master/juyan/result/sa_0712.npy")
np.save('/home/asdf/Documents/juyan/SSRN-master/juyan/result/sa_0712.npy', pre_test_pro)
collections.Counter(pre_test_pro)
# np.save('/home/asdf/Documents/juyan/SSRN-master/juyan/result/pavia.npy', pre_test_pro)
gt_test = gt[test_indices] - 1
overall_acc_res4 = metrics.accuracy_score(pre_test_pro[test_indices], gt_test)
confusion_matrix_res4 = metrics.confusion_matrix(pre_test_pro[test_indices], gt_test)
each_acc_res4, average_acc_res4 = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix_res4)
kappa = metrics.cohen_kappa_score(pre_test_pro[test_indices], gt_test)
KAPPA_RES_SS4.append(kappa)
OA_RES_SS4.append(overall_acc_res4)
AA_RES_SS4.append(average_acc_res4)

ELEMENT_ACC_RES_SS4[0, :] = each_acc_res4

print("3D RESNET_SS4 without BN training finished.")
print("# %d Iteration" % (index_iter + 1))

modelStatsRecord.outputStats(KAPPA_RES_SS4, OA_RES_SS4, AA_RES_SS4, ELEMENT_ACC_RES_SS4, CATEGORY,
                         '/home/asdf/Documents/juyan/SSRN-master/juyan/SA_train_SS_10.txt',
                         '/home/asdf/Documents/juyan/SSRN-master/juyan/SA_train_SS_element_10.txt')

# label_mat = np.zeros((512, 271), dtype=np.uint8)
# pre_label = pre_test_pro.argmax(axis=1)
# index = 0
# for ir in tqdm(range(whole_data.shape[0])):
#     for ic in range(whole_data.shape[1]):
#         label_mat[ir][ic] = pre_label[index]
#         index += 1
# np.save('/home/asdf/Documents/juyan/SSRN-master/juyan/result/SA_0712.npy', label_mat)


