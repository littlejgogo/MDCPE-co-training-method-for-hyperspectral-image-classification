import numpy as np
#
# labeled_index = np.load('/home/asdf/Documents/juyan/paper/co_training_code/newcode/dcpe_result'
#                                  '/data/cnn/unlabeled_index.npy')
# unlabeled_sets_rnn = np.load('/home/asdf/Documents/juyan/paper/co_training_code/newcode/dcpe_result'
#                                  '/data/cnn/labeled_index.npy')
#
# labeled_index2 = np.load('/home/asdf/Documents/juyan/paper/no_kmeans/data/rnn/unlabeled_index.npy')
# unlabeled_sets_rnn2 = np.load('/home/asdf/Documents/juyan/paper/no_kmeans/data/rnn/labeled_index.npy')
labeled_sets_rnn = np.load('/home/asdf/Documents/juyan/paper/paviau/cnn/data/labeled_index.npy')

test_sets_cnn = np.load('/home/asdf/Documents/juyan/paper/paviau/cnn/data/test_index.npy')
unlabeled_sets_cnn = np.load('/home/asdf/Documents/juyan/paper/paviau/cnn/data/unlabeled_index.npy')
valid_sets_cnn = np.load('/home/asdf/Documents/juyan/paper/paviau/cnn/data/valid_index.npy')
print("ok")