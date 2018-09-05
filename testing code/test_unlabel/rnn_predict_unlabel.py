import tensorflow as tf
import numpy as np
import rnn_indices

data = rnn_indices.read_data_sets()
batch_size = 2800
num_input = 12
timesteps = 17
unlabeled_sets = np.load('/home/asdf/Documents/juyan/paper/data/unlabeled_index.npy')
# num_steps = (len(unlabeled_sets) // batch_size) + 1


sess_rnn = tf.InteractiveSession()
new_saver_rnn = tf.train.import_meta_graph('/home/asdf/Documents/juyan/paper/co_training_code/newcode/model/rnn/pretrain_RNN.ckpt.meta')
new_saver_rnn.restore(sess_rnn, '/home/asdf/Documents/juyan/paper/co_training_code/newcode/model/rnn/pretrain_RNN.ckpt')

y_rnn = tf.get_collection('rnn_pred_label')[0]
graph_rnn = tf.get_default_graph()

X_rnn = graph_rnn.get_operation_by_name('X').outputs[0]
batch_x = data.unlabel
batch_x = batch_x.reshape((batch_size, timesteps, num_input))
pre_test = sess_rnn.run(y_rnn, feed_dict={X_rnn: batch_x})
pre_test = np.argmax(pre_test, axis=1) + 1
rnn_pre_test = pre_test.reshape((pre_test.shape[0], 1))
#sess_rnn.close().next_batch_test(batch_size, index)

