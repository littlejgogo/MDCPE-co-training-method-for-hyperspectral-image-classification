from sklearn import svm
import numpy as np
import final_index
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import rnn_indices
true_label = np.zeros((1, ), dtype=np.int32)
pre_label = np.zeros((1, ), dtype=np.int32)
data = rnn_indices.read_data_sets()
clf = svm.SVC(C=145., kernel='linear')
train_batch = data.train.num_examples
data_x, data_y = data.train.next_batch(train_batch)
clf.fit(data_x, data_y)
valid_batch = data.valid.num_examples
valid_data, valid_true = data.valid.next_batch(valid_batch)
valid_pre = clf.predict(valid_data)
ac = accuracy_score(valid_true, valid_pre)
print(ac)
# test_batch = 5000
# for index in range((data.test._num_examples // test_batch) + 1):
#     test_data, true_lab = data.test.next_batch_test(test_batch)
#     pre_lab = clf.predict(test_data)
#     true_label = np.concatenate((true_label, true_lab), axis=0)
#     pre_label = np.concatenate((pre_label, pre_lab), axis=0)
#
# true_label = true_label[1:]
# pre_label = pre_label[1:]
#
# every_class, confusion_mat = final_index.test_data_index(true_label, pre_label, 9)
# np.savez('/home/asdf/Documents/juyan/paper/paviau/svm-linear/test/test_zhibiao0522.npz',
#          every_class=every_class, confusion_mat=confusion_mat)
test_batch = 5000
for index in range((data.all._num_examples // test_batch) + 1):
    test_data = data.all.next_batch_testall(test_batch)
    pre_lab = clf.predict(test_data)
    # true_label = np.concatenate((true_label, true_lab), axis=0)
    pre_label = np.concatenate((pre_label, pre_lab), axis=0)

pre_label = pre_label[1:]
np.save("/home/asdf/Documents/juyan/paper/paviau/svm-linear/test/pre_alllabel.npy", pre_label)
print("ok")