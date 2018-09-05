import numpy as np
import scipy.io as sio
import scipy.misc
from tqdm import tqdm
colorbar = np.array([[0, 0, 255], [255, 0, 0], [0, 255, 0], [255, 255, 0], [0, 100, 0], [255, 0, 255], [0, 191, 255],
             [255, 140, 0], [255, 231, 186]])
each_class = np.load("/home/asdf/Documents/juyan/SSRN-master/juyan/result/pc_0712.npy")
each_class = each_class + 1
mat_gt = sio.loadmat("/home/asdf/Documents/juyan/paper/paviac/cnn/data/Pavia_gt.mat")
label = mat_gt['pavia_gt']

image = np.zeros((3, label.shape[0], label.shape[1]), dtype=np.int64)
groundtruth = np.zeros((3, label.shape[0], label.shape[1]), dtype=np.int64)
each_class_matrix = np.zeros((label.shape[0], label.shape[1]), dtype=np.int64)

index = 0
for ir in tqdm(range(label.shape[0])):
    for ic in range(label.shape[1]):
        each_class_matrix[ir][ic] = each_class[index]
        index += 1
# np.save('/home/asdf/Documents/juyan/SSRN-master/juyan/result/SA_0712.npy', label_mat)

for i in range(label.shape[0]):
    for j in range(label.shape[1]):
        for k in range(1, 10):
            if label[i, j] != 0 and each_class_matrix[i, j] == k:
                image[:, i, j] = colorbar[k - 1]

image = np.transpose(image, (1, 2, 0))

scipy.misc.imsave('/home/asdf/Documents/juyan/SSRN-master/juyan/result/pc.jpg', image)
# scipy.misc.imsave('/home/asdf/Documents/juyan/paper/paviau/cnn/test/groundtruth.jpg', groundtruth)
