# coding=utf-8
import skimage.io
import numpy as np
import time
import pickle
from sklearn.ensemble import RandomForestClassifier
import final_index
import rnn_indices
data = rnn_indices.read_data_sets()
true_label = np.zeros((1, ), dtype=np.int32)
pre_label = np.zeros((1, ), dtype=np.int32)

batch_size1 = data.train.num_examples
train_x, train_y = data.train.next_batch(batch_size1)
train_x = train_x[:600]
train_y = train_y[:600]
batch_size2 = data.valid.num_examples
valid_x, valid_y = data.valid.next_batch(batch_size2)
best = 0
# for i in range(20, 301):
rf = RandomForestClassifier(criterion="gini", max_features="sqrt",
                            n_estimators=50, min_samples_leaf=2, n_jobs=-1, oob_score=False)
rf.fit(train_x, train_y)
valid_acc = rf.score(valid_x, valid_y)
all_sets_index = np.load("/home/asdf/Documents/juyan/paper/paviac/cnn/data/all_index.npy")
# test_batch = 5000
# for index in range((data.test._num_examples // test_batch) + 1):
#     test_data, true_lab = data.test.next_batch_test(test_batch)
#     pre_lab = rf.predict(test_data)
#     true_label = np.concatenate((true_label, true_lab), axis=0)
#     pre_label = np.concatenate((pre_label, pre_lab), axis=0)
#
# true_label = true_label[1:]
# pre_label = pre_label[1:]
#
# every_class, confusion_mat = final_index.test_data_index(true_label, pre_label, 9)
# np.savez('/home/asdf/Documents/juyan/paper/paviac/rf/test/zhibiao0521.npz',
#          every_class=every_class, confusion_mat=confusion_mat)


test_batch = 5000
for index in range((data.all._num_examples // test_batch) + 1):
    test_data = data.all.next_batch_testall(test_batch)
    pre_lab = rf.predict(test_data)
    # true_label = np.concatenate((true_label, true_lab), axis=0)
    pre_label = np.concatenate((pre_label, pre_lab), axis=0)

pre_label = pre_label[1:]
np.save("/home/asdf/Documents/juyan/paper/paviac/rf/test/pre_alllabel.npy", pre_label)
print("ok")