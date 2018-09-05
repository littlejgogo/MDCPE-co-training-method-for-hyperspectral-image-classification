import tensorflow as tf
import rnn_indices
data = rnn_indices.read_data_sets()
import final_index
import numpy as np
num_input = 17  # MNIST data input (img shape: 28*28)
timesteps = 12
saver = tf.train.import_meta_graph('/home/asdf/Documents/juyan/paper/salinas/mdcpe_result/newmdcpe/model/2_rnn/'
                                   'RNN05072.ckpt.meta')
batch_size = 2000
# prediction = np.zeros((1, 16), dtype=np.int32)
# true_label = np.zeros((1, 16), dtype=np.int32)
rnnlogits = np.zeros((1, 16), dtype=np.float64)
with tf.Session() as sess:
    saver.restore(sess, '/home/asdf/Documents/juyan/paper/salinas/mdcpe_result/newmdcpe/model/2_rnn/'
                                   'RNN05072.ckpt')
    y = sess.graph.get_tensor_by_name('Softmax:0')
    X = sess.graph.get_operation_by_name('X').outputs[0]
    proba = sess.graph.get_tensor_by_name('add:0')
    for index in range((data.all._num_examples // batch_size)+1):
        batch = data.all.next_batch_test(batch_size)
        if index == (data.all._num_examples // batch_size):
            batch = batch.reshape(((data.all._num_examples % batch_size), timesteps, num_input))
        else:
            batch = batch.reshape((batch_size, timesteps, num_input))
        rnn_logits, pre_pro = sess.run([proba, y], feed_dict={X: batch})
        rnnlogits = np.concatenate((rnnlogits, rnn_logits), axis=0)
rnnlogits = rnnlogits[1:]
# np.save("/home/asdf/Documents/juyan/paper/salinas/dcpe_result/test/all2_rnnlogits.npy", rnnlogits)
np.save("/home/asdf/Documents/juyan/paper/salinas/mdcpe_result/test/all2_rnnlogits.npy", rnnlogits)

