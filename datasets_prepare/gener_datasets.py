import scipy.io as sio
import numpy as np

def dense_to_one_hot(labels_dense, num_classes=16):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()-1] = 1
    return labels_one_hot

def zeroPadding_3D(old_matrix, pad_length, pad_depth=0):
    new_matrix = np.lib.pad(old_matrix, ((pad_length, pad_length), (pad_length, pad_length), (pad_depth, pad_depth)), 'constant', constant_values=0)
    return new_matrix


def indexTovector(index_, Col, data):
    new_assign = []
    for counter, value in enumerate(index_):
        assign_0 = value // Col
        assign_1 = value % Col
        new_assign.append(data[assign_0, assign_1, :])#the index of sample
    # new_assign = np.ndarray(new_assign)
    return new_assign


def indexToAssignment(index_, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]#the index of sample
    return new_assign


def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len, pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch


def sampling(groundTruth):              #divide dataset into train and test datasets
    train = {}
    test = {}
    valid = {}
    train_indices = []
    valid_indices = []
    test_indices = []
    m = max(groundTruth)
    for o in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == o + 1]
        np.random.shuffle(indices)
        train[o] = indices[:120]
        valid[o] = indices[120:151]
        test[o] = indices
        train_indices += train[o]
        valid_indices += valid[o]
        test_indices += test[o]
    np.random.shuffle(train_indices)
    np.random.shuffle(valid_indices)
    np.random.shuffle(test_indices)
    return train_indices, valid_indices, test_indices

matpath = "/home/asdf/Documents/juyan/paper/data/Salinas_corrected.mat"
pcapath = "/home/asdf/Documents/juyan/paper/data/pca3_salinas.mat"
labelpath = "/home/asdf/Documents/juyan/paper/data/Salinas_gt.mat"
data_rnn = sio.loadmat(matpath)
data_cnn = sio.loadmat(pcapath)
label = sio.loadmat(labelpath)
num_classes = 16
rnndata = data_rnn.get('salinas_corrected')
cnndata = data_cnn.get('newdata')
readlabel = label.get('salinas_gt')
gt = readlabel.reshape(np.prod(readlabel.shape[:2]),)
train_indices, valid_indices, test_indices = sampling(gt)

y_train = gt[train_indices]
y_train = to_categorical(np.asarray(y_train))  # change to one-hot from

y_valid = gt[valid_indices]
y_valid = to_categorical(np.asarray(y_valid))

y_test = gt[test_indices]
y_test = to_categorical(np.asarray(y_test))
# rnn data processing
normdata = np.zeros((rnndata.shape[0], rnndata.shape[1], rnndata.shape[2]), dtype=np.float32)
for dim in range(rnndata.shape[2]):
    normdata[:, :, dim] = (rnndata[:, :, dim] - np.amin(rnndata[:, :, dim])) / \
                          float((np.amax(rnndata[:, :, dim]) - np.amin(rnndata[:, :, dim])))
xr_train = indexTovector(train_indices, rnndata.shape[1], normdata)
xr_valid = indexTovector(valid_indices, rnndata.shape[1], normdata)
xr_test = indexTovector(test_indices, rnndata.shape[1], normdata)
#cnn data processing
normcnn = np.zeros((cnndata.shape[0], cnndata.shape[1], cnndata.shape[2]), dtype=np.float32)
for dim in range(cnndata.shape[2]):
    normcnn[:, :, dim] = (cnndata[:, :, dim] - np.amin(cnndata[:, :, dim])) / \
                          float((np.amax(cnndata[:, :, dim]) - np.amin(cnndata[:, :, dim])))
PATCH_LENGTH = 8
padded_data = zeroPadding_3D(normcnn, PATCH_LENGTH)
train_assign = indexToAssignment(train_indices, cnndata.shape[1], PATCH_LENGTH)
xc_train = np.zeros((len(train_indices), 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, 3))
for i in range(len(train_assign)):
    xc_train[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)
valid_assign = indexToAssignment(valid_indices, cnndata.shape[1], PATCH_LENGTH)
xc_valid = np.zeros((len(valid_indices), 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, 3))
for i in range(len(valid_assign)):
    xc_valid[i] = selectNeighboringPatch(padded_data, valid_assign[i][0], valid_assign[i][1], PATCH_LENGTH)
test_assign = indexToAssignment(test_indices, cnndata.shape[1], PATCH_LENGTH)
xc_test = np.zeros((len(test_indices), 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, 3))
for i in range(len(test_assign)):
    xc_test[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)
np.savez('/home/asdf/Documents/juyan/paper/data/salinas_train.npz', rnn=xr_train, cnn=xc_train, label=y_train)
np.savez('/home/asdf/Documents/juyan/paper/data/salinas_valid.npz', rnn=xr_valid, cnn=xc_valid, label=y_valid)
np.savez('/home/asdf/Documents/juyan/paper/data/salinas_test.npz', rnn=xr_test, cnn=xc_test, label=y_test)
