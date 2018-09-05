import numpy as np
import scipy.io as sio
# unlabeled_sets = np.load('/home/asdf/Documents/juyan/paper/data/salinas/rnn/unlabeled_index.npy')
# labeled_sets = np.load('/home/asdf/Documents/juyan/paper/data/salinas/rnn/labeled_index.npy')
# # # mat_gt = sio.loadmat('/home/asdf/Documents/juyan/paper/data/salinas/cnn/Salinas_gt.mat')
# # GT = mat_gt['salinas_gt']
# print("ok")
#
a = np.load("/home/asdf/Documents/juyan/paper/paviac/mdcpe/contrastive model/0/zhibiao0511.npz")
bb = a["every_class"]
cc = a["confusion_mat"]
# b = np.load("/home/asdf/Documents/juyan/paper/salinas/mdcpe_result/contractive model/rnn3.npy")
print("ok")