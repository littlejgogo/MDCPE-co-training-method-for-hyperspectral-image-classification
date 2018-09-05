# coding=utf-8
import skimage.io
import numpy as np
import time
import pickle
from sklearn.ensemble import RandomForestClassifier
import rnn_indices
data = rnn_indices.read_data_sets()

batch_size1 = data.train.num_examples
train_x, train_y = data.train.next_batch(batch_size1)
batch_size2 = data.valid.num_examples
valid_x, valid_y = data.valid.next_batch(batch_size2)
best = 0.7142857142857143
for i in range(20, 301):
    rf = RandomForestClassifier(criterion="gini", max_features="sqrt",
                                n_estimators=54, min_samples_leaf=2, n_jobs=-1, oob_score=False)
    rf.fit(train_x, train_y)
    valid_acc = rf.score(valid_x, valid_y)
    if valid_acc > best:
        best = valid_acc
        with open("rfmodel.pkl", "wb") as f:
            pickle.dump(rf, f)
        print(valid_acc, i)
