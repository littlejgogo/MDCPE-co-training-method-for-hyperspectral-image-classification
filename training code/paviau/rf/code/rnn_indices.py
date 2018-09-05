from __future__ import print_function
import numpy as np
import scipy.io as sio

nb_classes = 9
INPUT_DIMENSION = 102


mat_gt = sio.loadmat("/home/asdf/Documents/juyan/paper/paviau/rnn/data/PaviaU_gt.mat")
GT = mat_gt['paviaU_gt']
GT = GT.reshape(np.prod(GT.shape[:2]),)
data_mat = sio.loadmat('/home/asdf/Documents/juyan/paper/paviau/rnn/data/pca_paviau.mat')
data_IN = data_mat['newdata']
normdata = np.zeros((data_IN.shape[0], data_IN.shape[1], data_IN.shape[2]), dtype=np.float32)
for dim in range(data_IN.shape[2]):
    normdata[:, :, dim] = (data_IN[:, :, dim] - np.amin(data_IN[:, :, dim])) / \
                          float((np.amax(data_IN[:, :, dim]) - np.amin(data_IN[:, :, dim])))

labeled_sets = np.load('/home/asdf/Documents/juyan/paper/paviau/rnn/data/labeled_index.npy')
test_sets = np.load('/home/asdf/Documents/juyan/paper/paviau/rnn/data/test_index.npy')
valid_sets = np.load('/home/asdf/Documents/juyan/paper/paviau/rnn/data/valid_index.npy')
# unlabeled_sets = np.load('/home/asdf/Documents/juyan/paper/original/data/unlabeled_index.npy')
all_sets = np.load("/home/asdf/Documents/juyan/paper/paviau/cnn/data/all_index.npy")

def dense_to_one_hot(labels_dense, num_classes=9):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()-1] = 1
    return labels_one_hot



class DataSet(object):
  def __init__(self, images):
    self._num_examples = images.shape[0]
    self._images = images
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    hsi_batch_patch = np.zeros((batch_size, INPUT_DIMENSION), dtype=np.float32)
    col = data_IN.shape[1]
    for q1 in range(batch_size):
      hsi_batch_patch[q1] = normdata[(self._images[start + q1] // col), (self._images[start + q1] % col), :]
    block = self._images[start:end]
    hsi_batch_label = GT[block]
    # hsi_batch_label = dense_to_one_hot(hsi_batch_label, num_classes=9)
    return hsi_batch_patch, hsi_batch_label
  #
  def next_batch_test(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
        self._index_in_epoch = self._num_examples
    end = self._index_in_epoch
    hsi_batch_patch = np.zeros((end-start, INPUT_DIMENSION), dtype=np.float32)
    col = data_IN.shape[1]
    for q1 in range(end - start):
        hsi_batch_patch[q1] = normdata[(self._images[start + q1] // col), (self._images[start + q1] % col)]
    block = self._images[start:end]
    hsi_batch_label = GT[block]
    # hsi_batch_label = dense_to_one_hot(hsi_batch_label, num_classes=9)
    return hsi_batch_patch, hsi_batch_label

  def next_batch_testall(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
        self._index_in_epoch = self._num_examples
    end = self._index_in_epoch
    hsi_batch_patch = np.zeros((end-start, INPUT_DIMENSION), dtype=np.float32)
    col = data_IN.shape[1]
    for q1 in range(end - start):
        hsi_batch_patch[q1] = normdata[(self._images[start + q1] // col), (self._images[start + q1] % col)]

    return hsi_batch_patch


def read_data_sets():
    class DataSets(object):
        pass
    data_sets = DataSets()
    data_sets.train = DataSet(labeled_sets)
    data_sets.valid = DataSet(valid_sets)
    # data_sets.unlabel = DataSet(unlabeled_sets)
    data_sets.test = DataSet(test_sets)
    data_sets.all = DataSet(all_sets)
    return data_sets

