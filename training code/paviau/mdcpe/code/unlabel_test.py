import tensorflow as tf
import math
import predict_unlabel
import scipy.io as sio
label = predict_unlabel.predict_models()
import numpy as np
import cnn_indices
data_cnn = cnn_indices.read_data_sets()
import rnn_indices
data_rnn = rnn_indices.read_data_sets()
import julei

import new_class_sample_update


batch_size = 5000
num_input = 17  # MNIST data input (img shape: 28*28)
timesteps = 6
logits_cnn = np.zeros((1, 9), dtype=np.float64)
prob_cnn = np.zeros((1, 9), dtype=np.float64)
logits_rnn = np.zeros((1, 9), dtype=np.float64)
prob_rnn = np.zeros((1, 9), dtype=np.float64)


for index1 in range((data_cnn.unlabel._num_examples // batch_size) + 1):
    batch_x1 = data_cnn.unlabel.next_batch_test(batch_size)
    logits_cnn_med, prob_cnn_med = label.cnn.predict1(batch_x1)
    logits_cnn = np.concatenate((logits_cnn, logits_cnn_med), axis=0)
    prob_cnn = np.concatenate((prob_cnn, prob_cnn_med), axis=0)

for index2 in range((data_rnn.unlabel._num_examples // batch_size) + 1):
    batch_x2 = data_rnn.unlabel.next_batch_test(batch_size)
    if index2 == (data_rnn.unlabel._num_examples // batch_size):
        batch_x2 = batch_x2.reshape(((data_rnn.unlabel._num_examples % batch_size), timesteps, num_input))
    else:
        batch_x2 = batch_x2.reshape((batch_size, timesteps, num_input))
    logits_rnn_med, prob_rnn_med = label.rnn.predict2(batch_x2)
    logits_rnn = np.concatenate((logits_rnn, logits_rnn_med), axis=0)
    prob_rnn = np.concatenate((prob_rnn, prob_rnn_med), axis=0)

prob_cnn = prob_cnn[1:]
prob_rnn = prob_rnn[1:]
logits_rnn = logits_rnn[1:]
logits_cnn = logits_cnn[1:]
# find same predicted label of two networks
label_cnn = np.argmax(prob_cnn, axis=1) + 1
label_rnn = np.argmax(prob_rnn, axis=1) + 1
same_indices = np.where(label_cnn == label_rnn)
big_dif = np.where(label_cnn != label_rnn)

batch_cnn_center = data_cnn.train.next_batch_test(284)

logits_cnn_center, prob_center_cnn = label.cnn.predict1(batch_cnn_center)
label_center_cnn = np.argmax(prob_center_cnn, axis=1) + 1
center_indices1 = []
juleicenter1 = {}
for i in range(9):
    indices = [j for j, x in enumerate(label_center_cnn.tolist()) if x == i+1]
    np.random.shuffle(indices)
    juleicenter1[i] = indices[:1]
    center_indices1 += juleicenter1[i]
cnn_center_data = logits_cnn_center[center_indices1]
logits_cnn = np.concatenate((cnn_center_data, logits_cnn), axis=0)

batch_rnn_center = data_rnn.train.next_batch_test(284)
batch_rnn_center = batch_rnn_center.reshape((284, timesteps, num_input))
logits_rnn_center, prob_center_rnn = label.rnn.predict2(batch_rnn_center)
label_center_rnn = np.argmax(prob_center_rnn, axis=1) + 1
center_indices2 = []
juleicenter2 = {}
for i in range(9):
    indices = [j for j, x in enumerate(label_center_rnn.tolist()) if x == i+1]
    np.random.shuffle(indices)
    juleicenter2[i] = indices[:1]
    center_indices2 += juleicenter2[i]
rnn_center_data = logits_rnn_center[center_indices2]
logits_rnn = np.concatenate((rnn_center_data, logits_rnn), axis=0)

norm_rnn = np.zeros((logits_rnn.shape[0], logits_rnn.shape[1]), dtype=np.float32)
norm_cnn = np.zeros((logits_cnn.shape[0], logits_cnn.shape[1]), dtype=np.float32)
max_cnn = np.amax(logits_cnn, axis=1)
min_cnn = np.amin(logits_cnn, axis=1)
substract_cnn = [x-y for x, y in zip(max_cnn, min_cnn)]
max_rnn = np.amax(logits_rnn, axis=1)
min_rnn = np.amin(logits_rnn, axis=1)
substract_rnn = [x-y for x, y in zip(max_rnn, min_rnn)]
for i in range(logits_cnn.shape[0]):
    for j in range(logits_cnn.shape[1]):
        norm_cnn[i][j] = (logits_cnn[i][j] - min_cnn[i]) / substract_cnn[i]
        norm_rnn[i][j] = (logits_rnn[i][j] - min_rnn[i]) / substract_rnn[i]
cnn_center = norm_cnn[:9]
cnn_unlabeled = norm_cnn[9:]
rnn_center = norm_rnn[:9]
rnn_unlabeled = norm_rnn[9:]

def update_cnn():
    max_diff_cnn = [x-y for x, y in zip(rnn_unlabeled, cnn_unlabeled)]
    max_dcpe_cnn = np.argmax(max_diff_cnn, axis=1) + 1
    remove_same_cnn = (big_dif[0]).tolist()
    np.random.shuffle(remove_same_cnn)
    same_indexall_cnn = (same_indices[0]).tolist()
    np.random.shuffle(same_indexall_cnn)
    print("kmeans and cnn same samples begin")
    kmeannetcnnsame_indices = julei.predict(cnn_unlabeled[same_indexall_cnn], cnn_center,
                                            label_cnn[same_indexall_cnn], same_indexall_cnn)
    same_cnn_index = new_class_sample_update.update1(kmeannetcnnsame_indices, label_cnn, same_indexall_cnn)
    # same_cnn_index = new_class_sample_update.update(same_indexall_cnn, label_cnn)

    print("kmeans and cnn dcpe samples begin")
    #
    kmeannetcnndcpe_indices = julei.predict(cnn_unlabeled[remove_same_cnn], cnn_center,
                                            max_dcpe_cnn[remove_same_cnn], remove_same_cnn)
    dcpe_cnn_index = new_class_sample_update.update2(kmeannetcnndcpe_indices, max_dcpe_cnn, remove_same_cnn)
    # dcpe_cnn_index = new_class_sample_update.update(remove_same_cnn, max_dcpe_cnn)
    cnn_index_all = same_cnn_index + dcpe_cnn_index
    cnn_label_all = np.concatenate((label_cnn[same_cnn_index], max_dcpe_cnn[dcpe_cnn_index]), axis=0)

    labeled_sets_cnn = np.load('/home/asdf/Documents/juyan/paper/paviau/mdcpe/newmdcpe/data/cnn/labeled_index.npy')
    unlabeled_sets_cnn = np.load('/home/asdf/Documents/juyan/paper/paviau/mdcpe/newmdcpe/data/cnn/unlabeled_index.npy')
    mat_gt_cnn = sio.loadmat("/home/asdf/Documents/juyan/paper/paviau/mdcpe/newmdcpe/data/cnn/PaviaU_gt.mat")
    GT_cnn = mat_gt_cnn['paviaU_gt']
    cnn_index_bigmap = unlabeled_sets_cnn[cnn_index_all]
    update_labeled_sets_cnn = np.concatenate((labeled_sets_cnn, cnn_index_bigmap), axis=0)
    gt_col_cnn = GT_cnn.reshape((GT_cnn.shape[0]*GT_cnn.shape[1], ))
    cnn_index_bigmap = cnn_index_bigmap.tolist()
    gt_col_cnn[cnn_index_bigmap] = cnn_label_all
    real_gt_cnn = gt_col_cnn.reshape((GT_cnn.shape[0], GT_cnn.shape[1]))
    # unlabeled_sets_cnn = list(set(unlabeled_sets_cnn.tolist()).difference(set(cnn_index_bigmap)))
    np.random.shuffle(update_labeled_sets_cnn)
    np.save('/home/asdf/Documents/juyan/paper/paviau/mdcpe/newmdcpe/data/cnn/unlabeled_index.npy', unlabeled_sets_cnn)
    np.save('/home/asdf/Documents/juyan/paper/paviau/mdcpe/newmdcpe/data/cnn/labeled_index.npy', update_labeled_sets_cnn)
    sio.savemat('/home/asdf/Documents/juyan/paper/paviau/mdcpe/newmdcpe/data/cnn/PaviaU_gt.mat', {'paviaU_gt': real_gt_cnn})
    return ()


def update_rnn():
    max_diff_rnn = [x - y for x, y in zip(cnn_unlabeled, rnn_unlabeled)]
    max_dcpe_rnn = np.argmax(max_diff_rnn, axis=1) + 1
    remove_same_rnn = (big_dif[0]).tolist()
    np.random.shuffle(remove_same_rnn)
    same_indexall_rnn = (same_indices[0]).tolist()
    np.random.shuffle(same_indexall_rnn)
    print("kmeans and rnn same samples begin")
    kmeannetrnnsame_indices = julei.predict(rnn_unlabeled[same_indexall_rnn], rnn_center,
                                         label_rnn[same_indexall_rnn], same_indexall_rnn)
    same_rnn_index = new_class_sample_update.update1(kmeannetrnnsame_indices, label_rnn, same_indexall_rnn)
    # same_rnn_index = new_class_sample_update.update(same_indexall_rnn, label_rnn)
    print("kmeans and rnn dcpe samples begin")
    kmeannetrnndcpe_indices = julei.predict(rnn_unlabeled[remove_same_rnn], rnn_center,
                                             max_dcpe_rnn[remove_same_rnn], remove_same_rnn)
    dcpe_rnn_index = new_class_sample_update.update2(kmeannetrnndcpe_indices, max_dcpe_rnn, remove_same_rnn)
    # dcpe_rnn_index = new_class_sample_update.update(remove_same_rnn, max_dcpe_rnn)
    rnn_index_all = same_rnn_index + dcpe_rnn_index
    rnn_label_all = np.concatenate((label_rnn[same_rnn_index], max_dcpe_rnn[dcpe_rnn_index]), axis=0)

    labeled_sets_rnn = np.load('/home/asdf/Documents/juyan/paper/paviau/mdcpe/newmdcpe/data/rnn/labeled_index.npy')
    unlabeled_sets_rnn = np.load('/home/asdf/Documents/juyan/paper/paviau/mdcpe/newmdcpe/data/rnn/unlabeled_index.npy')
    mat_gt_rnn = sio.loadmat("/home/asdf/Documents/juyan/paper/paviau/mdcpe/newmdcpe/data/rnn/PaviaU_gt.mat")
    GT_rnn = mat_gt_rnn['paviaU_gt']
    rnn_index_bigmap = unlabeled_sets_rnn[rnn_index_all]
    update_labeled_sets_rnn = np.concatenate((labeled_sets_rnn, rnn_index_bigmap), axis=0)
    gt_col_rnn = GT_rnn.reshape((GT_rnn.shape[0] * GT_rnn.shape[1],))
    gt_col_rnn[rnn_index_bigmap] = rnn_label_all
    real_gt_rnn = gt_col_rnn.reshape((GT_rnn.shape[0], GT_rnn.shape[1]))
    # unlabeled_sets_rnn = list(set(unlabeled_sets_rnn.tolist()).difference(set(rnn_index_bigmap)))
    np.random.shuffle(update_labeled_sets_rnn)
    np.save('/home/asdf/Documents/juyan/paper/paviau/mdcpe/newmdcpe/data/rnn/unlabeled_index.npy', unlabeled_sets_rnn)
    np.save('/home/asdf/Documents/juyan/paper/paviau/mdcpe/newmdcpe/data/rnn/labeled_index.npy', update_labeled_sets_rnn)
    sio.savemat('/home/asdf/Documents/juyan/paper/paviau/mdcpe/newmdcpe/data/rnn/PaviaU_gt.mat', {'paviaU_gt': real_gt_rnn})
    return ()


update_cnn()
update_rnn()




