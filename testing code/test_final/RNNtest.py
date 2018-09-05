import tensorflow as tf
import rnn_indices
data = rnn_indices.read_data_sets()
import final_index
import numpy as np
timesteps = 17
num_input = 12
saver = tf.train.import_meta_graph('/home/asdf/Documents/juyan/paper/co_training_code/newcode/041815eachclass/'
                                   'model/0_rnn/pretrain_RNN0418.ckpt.meta')
batch_size = 2000
prediction = np.zeros((1, 16), dtype=np.int32)
true_label = np.zeros((1, 16), dtype=np.int32)
rnnlogits = np.zeros((1, 16), dtype=np.float64)
with tf.Session() as sess:
    saver.restore(sess, '/home/asdf/Documents/juyan/paper/co_training_code/newcode/041815eachclass/'
                        'model/0_rnn/pretrain_RNN0418.ckpt')
    y = sess.graph.get_tensor_by_name('Softmax:0')
    X = sess.graph.get_operation_by_name('X').outputs[0]
    proba = sess.graph.get_tensor_by_name('add:0')
    # batch = batch.reshape((batch_size, timesteps, num_input))

    # keep_prob = sess.graph.get_operation_by_name('keep_prob').outputs[0]
    for index in range((data.test._num_examples // batch_size)+1):
        batch, Y = data.test.next_batch_test(batch_size)
        if index == (data.test._num_examples // batch_size):
            batch = batch.reshape(((data.test._num_examples % batch_size), timesteps, num_input))
        else:
            batch = batch.reshape((batch_size, timesteps, num_input))
        rnn_logits, pre_pro = sess.run([proba, y], feed_dict={X: batch})
        # prediction = np.concatenate((prediction, predict), axis=0)
        # true_label = np.concatenate((true_label, Y), axis=0)
        rnnlogits = np.concatenate((rnnlogits, rnn_logits), axis=0)
# predict_label = np.argmax(prediction[1:], 1) + 1
# true_label = np.argmax(true_label[1:], 1) + 1
rnnlogits = rnnlogits[1:]
np.save("/home/asdf/Documents/juyan/paper/co_training_code/newcode/test_final/result/0_rnnresult.npy", rnnlogits)
# every_class, confusion_mat = final_index.test_data_index(true_label, predict_label, 16)
# np.savez('/home/asdf/Documents/juyan/paper/data/salinas/0418_15each_class/zhibiao0421_rnn15.npz',
#          every_class=every_class, confusion_mat=confusion_mat)
# print("ok")

