import tensorflow as tf
import rnn_indices
data = rnn_indices.read_data_sets()
import final_index
import numpy as np
num_input = 17  # MNIST data input (img shape: 28*28)
timesteps = 6
saver = tf.train.import_meta_graph('/home/asdf/Documents/juyan/paper/paviac/mdcpe/newmdcpe/model/2_rnn/'
                                   'RNN05112.ckpt.meta')
batch_size = 5000
# prediction = np.zeros((1, 9), dtype=np.int32)
# true_label = np.zeros((1, 9), dtype=np.int32)
rnnlogits = np.zeros((1, 9), dtype=np.float64)
with tf.Session() as sess:
    saver.restore(sess, '/home/asdf/Documents/juyan/paper/paviac/mdcpe/newmdcpe/model/2_rnn/'
                                    'RNN05112.ckpt')
    y = sess.graph.get_tensor_by_name('Softmax:0')
    X = sess.graph.get_operation_by_name('X').outputs[0]
    proba = sess.graph.get_tensor_by_name('add:0')
    for index in range((data.all._num_examples // batch_size)+1):
        batch, Y = data.all.next_batch_test(batch_size)
        if index == (data.all._num_examples // batch_size):
            batch = batch.reshape(((data.all._num_examples % batch_size), timesteps, num_input))
        else:
            batch = batch.reshape((batch_size, timesteps, num_input))
        rnn_logits, pre_pro = sess.run([proba, y], feed_dict={X: batch})
        # prediction = np.concatenate((prediction, pre_pro), axis=0)
        # true_label = np.concatenate((true_label, Y), axis=0)
        rnnlogits = np.concatenate((rnnlogits, rnn_logits), axis=0)
# predict_label = np.argmax(prediction[1:], 1) + 1
# true_label = np.argmax(true_label[1:], 1) + 1
rnnlogits = rnnlogits[1:]
np.save("/home/asdf/Documents/juyan/paper/paviac/mdcpe/testresult/all_logits.npy", rnnlogits)
# every_class, confusion_mat = final_index.test_data_index(true_label, predict_label, 9)
# np.savez('/home/asdf/Documents/juyan/paper/paviac/mdcpe/testresult/rnn/zhibiao0517.npz',
#          every_class=every_class, confusion_mat=confusion_mat)
print("ok")
#
# import tensorflow as tf
# import rnn_indices
# data = rnn_indices.read_data_sets()
# import final_index
# import numpy as np
# num_input = 17  # MNIST data input (img shape: 28*28)
# timesteps = 6
# saver = tf.train.import_meta_graph('/home/asdf/Documents/juyan/paper/paviac/mdcpe/newmdcpe/model/9_rnn/'
#                                    'RNN05119.ckpt.meta')
# batch_size = data.valid._num_examples
# prediction = np.zeros((1, 9), dtype=np.int32)
# true_label = np.zeros((1, 9), dtype=np.int32)
# rnnlogits = np.zeros((1, 9), dtype=np.float64)
# with tf.Session() as sess:
#     saver.restore(sess, '/home/asdf/Documents/juyan/paper/paviac/mdcpe/newmdcpe/model/10_rnn/'
#                                    'RNN051110.ckpt')
#     y = sess.graph.get_tensor_by_name('Softmax:0')
#     X = sess.graph.get_operation_by_name('X').outputs[0]
#     proba = sess.graph.get_tensor_by_name('add:0')
#     # batch = batch.reshape((batch_size, timesteps, num_input))
#
#     # keep_prob = sess.graph.get_operation_by_name('keep_prob').outputs[0]
#
#     batch, Y = data.valid.next_batch_test(batch_size)
#     batch = batch.reshape((batch_size, timesteps, num_input))
#     rnn_logits, pre_pro = sess.run([proba, y], feed_dict={X: batch})
#     # prediction = np.concatenate((prediction, predict), axis=0)
#     # true_label = np.concatenate((true_label, Y), axis=0)
#     rnnlogits = np.concatenate((rnnlogits, rnn_logits), axis=0)
# # predict_label = np.argmax(prediction[1:], 1) + 1
# # true_label = np.argmax(true_label[1:], 1) + 1
# rnnlogits = rnnlogits[1:]
# np.save("/home/asdf/Documents/juyan/paper/paviac/mdcpe/newmdcpe/model/10_rnn/logits.npy", rnnlogits)
# # every_class, confusion_mat = final_index.test_data_index(true_label, predict_label, 16)
# # np.savez('/home/asdf/Documents/juyan/paper/data/salinas/0418_15each_class/zhibiao0421_rnn15.npz',
# #          every_class=every_class, confusion_mat=confusion_mat)
# # print("ok")

