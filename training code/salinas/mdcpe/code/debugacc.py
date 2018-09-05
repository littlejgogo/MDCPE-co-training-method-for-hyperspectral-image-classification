import numpy as np
import scipy.io as sio
# unlabeled_sets = np.load('/home/asdf/Documents/juyan/paper/data/salinas/rnn/unlabeled_index.npy')
# labeled_sets = np.load('/home/asdf/Documents/juyan/paper/data/salinas/rnn/labeled_index.npy')
# # # mat_gt = sio.loadmat('/home/asdf/Documents/juyan/paper/data/salinas/cnn/Salinas_gt.mat')
# # GT = mat_gt['salinas_gt']
# print("ok")
#
a = np.load("/home/asdf/Documents/juyan/paper/salinas/mdcpe_result/newmdcpe/cnn10.npy")
b = np.load("/home/asdf/Documents/juyan/paper/salinas/mdcpe_result/newmdcpe/rnn10.npy")

cnn = np.load("/home/asdf/Documents/juyan/paper/salinas/mdcpe_result/model/allcnn_acc10.npy")
rnn = np.load("/home/asdf/Documents/juyan/paper/salinas/mdcpe_result/model/allrnn_acc10.npy")
# a = np.load("/home/asdf/Documents/juyan/paper/salinas/mdcpe_result/contractive model/0/zhibiao0513.npz")
# oa = a["every_class"]
print("ok")