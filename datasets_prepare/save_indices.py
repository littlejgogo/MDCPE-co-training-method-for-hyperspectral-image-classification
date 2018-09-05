# # -*- coding: utf-8 -*-
# import numpy as np
#
# import scipy.io as sio
#
#
# def sampling(groundTruth):              #divide dataset into train and test datasets
#     labeled = {}
#     test = {}
#     valid = {}
#     m = max(groundTruth)
#     labeled_indices = []
#     test_indices = []
#     valid_indices = []
#     unlabeled_indices = []
#
#     for i in range(m+1):
#         indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i]
#         if i == 0:
#             np.random.shuffle(indices)
#             unlabeled_indices = indices
#         else:
#             if i == 4 or i == 9 or i ==3:
#                 np.random.shuffle(indices)
#                 test[i] = indices[70:]
#                 valid[i] = indices[17:70]
#                 labeled[i] = indices[:17]
#                 labeled_indices += labeled[i]
#                 test_indices += test[i]
#                 valid_indices += valid[i]
#             else:
#                 if i == 1 or i == 8:
#                     np.random.shuffle(indices)
#                     test[i] = indices[1000:]
#                     valid[i] = indices[285:1000]
#                     labeled[i] = indices[:285]
#                     labeled_indices += labeled[i]
#                     test_indices += test[i]
#                     valid_indices += valid[i]
#                 else:
#                     np.random.shuffle(indices)
#                     test[i] = indices[170:]
#                     valid[i] = indices[42:170]
#                     labeled[i] = indices[:42]
#                     labeled_indices += labeled[i]
#                     test_indices += test[i]
#                     valid_indices += valid[i]
#     np.random.shuffle(labeled_indices)
#     np.random.shuffle(test_indices)
#     np.random.shuffle(valid_indices)
#
#     return labeled_indices, test_indices, valid_indices, unlabeled_indices
#
#
# mat_gt = sio.loadmat("/home/asdf/Documents/juyan/paper/pavia/Pavia_gt.mat")
# gt_IN = mat_gt['pavia_gt']
# new_gt_IN = gt_IN
#
# gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]),)
#
#
# labeled_indices, test_indices, valid_indices, unlabeled_indices = sampling(gt)
#
# print(len(labeled_indices))
# print(len(valid_indices))
# print(len(unlabeled_indices))
# print(len(test_indices))
#
# np.save('/home/asdf/Documents/juyan/paper/pavia/labeled_index.npy', labeled_indices)
# np.save('/home/asdf/Documents/juyan/paper/pavia/valid_index.npy', valid_indices)
# np.save('/home/asdf/Documents/juyan/paper/pavia/unlabeled_index.npy', unlabeled_indices)
# np.save('/home/asdf/Documents/juyan/paper/pavia/test_index.npy', test_indices)
#
#
# -*- coding: utf-8 -*-
import numpy as np

import scipy.io as sio


def sampling(groundTruth):              #divide dataset into train and test datasets
    all = {}
    m = max(groundTruth)
    all_indices = []
    for i in range(m+1):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i]
        if i != 0:
            np.random.shuffle(indices)
            all[i] = indices
            all_indices += all[i]
    np.random.shuffle(all_indices)
    return all_indices


mat_gt = sio.loadmat("/home/asdf/Documents/juyan/paper/salinas/cnn/data/Salinas_gt.mat")
gt_IN = mat_gt['salinas_gt']
new_gt_IN = gt_IN
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]),)
all_indices = sampling(gt)

print(len(all_indices))

np.save('/home/asdf/Documents/juyan/paper/salinas/cnn/data/all_index.npy', all_indices)







