import numpy as np
import scipy.io as sio
import scipy.misc
import cnn_indices
data = cnn_indices.read_data_sets()
batch_size = data.train._num_examples
train_label = data.train.next_batch_test(batch_size)
all_sets_index = np.load("/home/asdf/Documents/juyan/paper/paviau/cnn/data/labeled_index.npy")
# colorbar = np.array([[255, 105, 180], [255, 0, 255], [147, 112, 219], [0, 0, 255], [25, 25, 112], [100, 149, 237], [0, 191, 255],
#             [0, 255, 0], [128, 0, 128], [85, 107, 47], [128, 128, 0], [255, 215, 0], [255, 140, 0],
#             [112, 128, 144], [128, 0, 0], [0, 0, 0], [255, 255, 255]])
colorbar = np.array([[0, 0, 255], [255, 0, 0], [0, 255, 0], [255, 255, 0], [0, 100, 0], [255, 0, 255], [0, 191, 255],
             [255, 140, 0], [255, 231, 186]])
# each_class = np.load("/home/asdf/Documents/juyan/paper/paviau/svm-linear/test/pre_alllabel.npy")
each_class = train_label
mat_gt = sio.loadmat("/home/asdf/Documents/juyan/paper/paviau/cnn/data/PaviaU_gt.mat")
label = mat_gt['paviaU_gt']

image = np.zeros((3, label.shape[0], label.shape[1]), dtype=np.int64)
groundtruth = np.zeros((3, label.shape[0], label.shape[1]), dtype=np.int64)

for i in range(len(all_sets_index)):
    row = all_sets_index[i] // label.shape[1]
    col = all_sets_index[i] % label.shape[1]
    for k in range(1, 10):
        if label[row, col] == k:
            groundtruth[:, row, col] = colorbar[k-1]
        if each_class[i] == k:
            image[:, row, col] = colorbar[k-1]

image = np.transpose(image, (1, 2, 0))
groundtruth = np.transpose(groundtruth, (1, 2, 0))
scipy.misc.imsave('/home/asdf/Documents/juyan/paper/paviau/cnn/test/train_result.jpg', image)
# scipy.misc.imsave('/home/asdf/Documents/juyan/paper/paviau/cnn/test/groundtruth.jpg', groundtruth)
