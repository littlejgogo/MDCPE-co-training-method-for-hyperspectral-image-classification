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

batch_size = 2800
num_input = 17  # MNIST data input (img shape: 28*28)
timesteps = 12

batch_x1 = data_cnn.unlabel.next_batch_test(batch_size)
logits_cnn, prob_cnn = label.cnn.predict1(batch_x1)


batch_x2 = data_rnn.unlabel.next_batch_test(batch_size)
batch_x2 = batch_x2.reshape((batch_size, timesteps, num_input))
logits_rnn, prob_rnn = label.rnn.predict2(batch_x2)

# find same predicted label of two networks
label_cnn = np.argmax(prob_cnn, axis=1) + 1
label_rnn = np.argmax(prob_rnn, axis=1) + 1
same_indices = np.where(label_cnn == label_rnn)
big_dif = np.where(label_cnn != label_rnn)


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

cnn_unlabeled = norm_cnn
rnn_unlabeled = norm_rnn


def update_cnn():
    max_diff_cnn = [x-y for x, y in zip(rnn_unlabeled, cnn_unlabeled)]
    max_dcpe_cnn = np.argmax(max_diff_cnn, axis=1) + 1
    remove_same_cnn = (big_dif[0]).tolist()
    np.random.shuffle(remove_same_cnn)
    same_indexall_cnn = (same_indices[0]).tolist()
    np.random.shuffle(same_indexall_cnn)
    same_cnn_index = same_indexall_cnn[:60]
    dcpe_cnn_index = remove_same_cnn[:60]
    cnn_index_all = same_cnn_index + dcpe_cnn_index
    cnn_label_all = np.concatenate((label_cnn[same_cnn_index], max_dcpe_cnn[dcpe_cnn_index]), axis=0)

    unlabeled_sets_cnn = np.load('/home/asdf/Documents/juyan/paper/salinas/dcpe_result/data/cnn/unlabeled_index.npy')
    labeled_sets_cnn = np.load('/home/asdf/Documents/juyan/paper/salinas/dcpe_result/data/cnn/labeled_index.npy')
    mat_gt_cnn = sio.loadmat('/home/asdf/Documents/juyan/paper/salinas/dcpe_result/data/cnn/Salinas_gt.mat')
    GT_cnn = mat_gt_cnn['salinas_gt']
    cnn_index_bigmap = (unlabeled_sets_cnn[:2800])[cnn_index_all]
    update_labeled_sets_cnn = np.concatenate((labeled_sets_cnn, cnn_index_bigmap), axis=0)
    gt_col_cnn = GT_cnn.reshape((GT_cnn.shape[0]*GT_cnn.shape[1], ))
    cnn_index_bigmap = cnn_index_bigmap.tolist()
    gt_col_cnn[cnn_index_bigmap] = cnn_label_all
    real_gt_cnn = gt_col_cnn.reshape((GT_cnn.shape[0], GT_cnn.shape[1]))
    # unlabeled_sets_cnn = list(set(unlabeled_sets_cnn.tolist()).difference(set(cnn_index_bigmap)))
    np.random.shuffle(update_labeled_sets_cnn)
    unlabeled_sets_cnn = unlabeled_sets_cnn[2800:]
    np.save('/home/asdf/Documents/juyan/paper/salinas/dcpe_result/data/cnn/unlabeled_index.npy', unlabeled_sets_cnn)
    np.save('/home/asdf/Documents/juyan/paper/salinas/dcpe_result/data/cnn/labeled_index.npy', update_labeled_sets_cnn)
    sio.savemat('/home/asdf/Documents/juyan/paper/salinas/dcpe_result/data/cnn/Salinas_gt.mat', {'salinas_gt': real_gt_cnn})
    return ()


def update_rnn():
    max_diff_rnn = [x - y for x, y in zip(cnn_unlabeled, rnn_unlabeled)]
    max_dcpe_rnn = np.argmax(max_diff_rnn, axis=1) + 1
    remove_same_rnn = (big_dif[0]).tolist()
    np.random.shuffle(remove_same_rnn)
    same_indexall_rnn = (same_indices[0]).tolist()
    np.random.shuffle(same_indexall_rnn)
    same_rnn_index = same_indexall_rnn[:60]
    dcpe_rnn_index = remove_same_rnn[:60]
    rnn_index_all = same_rnn_index + dcpe_rnn_index
    rnn_label_all = np.concatenate((label_rnn[same_rnn_index], max_dcpe_rnn[dcpe_rnn_index]), axis=0)

    unlabeled_sets_rnn = np.load('/home/asdf/Documents/juyan/paper/salinas/dcpe_result/data/rnn/unlabeled_index.npy')
    labeled_sets_rnn = np.load('/home/asdf/Documents/juyan/paper/salinas/dcpe_result/data/rnn/labeled_index.npy')
    mat_gt_rnn = sio.loadmat('/home/asdf/Documents/juyan/paper/salinas/dcpe_result/data/rnn/Salinas_gt.mat')
    GT_rnn = mat_gt_rnn['salinas_gt']
    rnn_index_bigmap = (unlabeled_sets_rnn[:2800])[rnn_index_all]
    update_labeled_sets_rnn = np.concatenate((labeled_sets_rnn, rnn_index_bigmap), axis=0)
    gt_col_rnn = GT_rnn.reshape((GT_rnn.shape[0] * GT_rnn.shape[1],))
    gt_col_rnn[rnn_index_bigmap] = rnn_label_all
    real_gt_rnn = gt_col_rnn.reshape((GT_rnn.shape[0], GT_rnn.shape[1]))
    unlabeled_sets_rnn = unlabeled_sets_rnn[2800:]
    np.random.shuffle(update_labeled_sets_rnn)
    np.save('/home/asdf/Documents/juyan/paper/salinas/dcpe_result/data/rnn/unlabeled_index.npy',
            unlabeled_sets_rnn)
    np.save('/home/asdf/Documents/juyan/paper/salinas/dcpe_result/data/rnn/labeled_index.npy',
            update_labeled_sets_rnn)
    sio.savemat('/home/asdf/Documents/juyan/paper/salinas/dcpe_result/data/rnn/Salinas_gt.mat',
                {'salinas_gt': real_gt_rnn})
    return ()


update_cnn()
update_rnn()




