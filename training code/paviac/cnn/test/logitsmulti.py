
import tensorflow as tf
import cnn_indices

data = cnn_indices.read_data_sets()
import final_index
import numpy as np
saver = tf.train.import_meta_graph('/home/asdf/Documents/juyan/paper/paviac/cnn/model/'
                                   'CNN0511.ckpt.meta')
batch_size = 5000
prediction = np.zeros((1, 9), dtype=np.int32)
# true_label = np.zeros((1, 9), dtype=np.int32)
with tf.Session() as sess:
    saver.restore(sess, '/home/asdf/Documents/juyan/paper/paviac/cnn/model/'
                                   'CNN0511.ckpt')
    y = sess.graph.get_tensor_by_name('Softmax:0')
    X = sess.graph.get_operation_by_name('X').outputs[0]
    keep_prob = sess.graph.get_operation_by_name('keep_prob').outputs[0]
    for index in range((data.all._num_examples // batch_size) + 1):
        batch = data.all.next_batch_test(batch_size)
        predict_label = sess.run(y, feed_dict={X: batch, keep_prob: 1.0})
        prediction = np.concatenate((prediction, predict_label), axis=0)
        # true_label = np.concatenate((true_label, Y), axis=0)
pred_label = np.argmax(prediction[1:], 1) + 1
# truelabel = np.argmax(true_label[1:], 1) + 1
# every_class, confusion_mat = final_index.test_data_index(truelabel, pred_label, 9)
# np.savez('/home/asdf/Documents/juyan/paper/paviac/cnn/test/zhibiao0513test.npz',
#          every_class=every_class, confusion_mat=confusion_mat)
np.save("/home/asdf/Documents/juyan/paper/paviac/cnn/test/pre_alllabel.npy", pred_label)



# with tf.Session() as sess:
#     saver.restore(sess, '/home/asdf/Documents/juyan/paper/paviac/mdcpe/newmdcpe/model/10_cnn/'
#                                    'CNN051110.ckpt')
#     y = sess.graph.get_tensor_by_name('Softmax:0')
#     X = sess.graph.get_operation_by_name('X').outputs[0]
#     keep_prob = sess.graph.get_operation_by_name('keep_prob').outputs[0]
#     proba = sess.graph.get_tensor_by_name('Add_1:0')
#     for index in range((data.test._num_examples // batch_size) + 1):
#         batch, Y = data.test.next_batch_test(batch_size)
#         cnn_logits, pre_pro = sess.run([proba, y], feed_dict={X: batch, keep_prob: 1.0})
#         # prediction = np.concatenate((prediction, pre_pro), axis=0)
#         true_label = np.concatenate((true_label, Y), axis=0)
#         cnnlogits = np.concatenate((cnnlogits, cnn_logits), axis=0)
# # predict_label = np.argmax(prediction[1:], 1) + 1
# true_label = np.argmax(true_label[1:], 1) + 1
# # prediction = prediction[1:]
# cnnlogits = cnnlogits[1:]
# rnnlogtis = np.load("/home/asdf/Documents/juyan/paper/paviac/mdcpe/testresult/logits10.npy")
#
# norm_rnn = np.zeros((cnnlogits.shape[0], cnnlogits.shape[1]), dtype=np.float32)
# norm_cnn = np.zeros((cnnlogits.shape[0], cnnlogits.shape[1]), dtype=np.float32)
# max_cnn = np.amax(cnnlogits, axis=1)
# min_cnn = np.amin(cnnlogits, axis=1)
# substract_cnn = [x-y for x, y in zip(max_cnn, min_cnn)]
# max_rnn = np.amax(rnnlogtis, axis=1)
# min_rnn = np.amin(rnnlogtis, axis=1)
# substract_rnn = [x-y for x, y in zip(max_rnn, min_rnn)]
# for i in range(cnnlogits.shape[0]):
#     for j in range(cnnlogits.shape[1]):
#         norm_cnn[i][j] = (cnnlogits[i][j] - min_cnn[i]) / substract_cnn[i]
#         norm_rnn[i][j] = (rnnlogtis[i][j] - min_rnn[i]) / substract_rnn[i]
#
#
# alllogits = [x * y for x, y in zip(norm_cnn, norm_rnn)]
#
# predict_label = np.argmax(alllogits, 1) + 1
#
# every_class, confusion_mat = final_index.test_data_index(true_label, predict_label, 9)
# np.savez('/home/asdf/Documents/juyan/paper/paviac/mdcpe/testresult/zhibiao10.npz',
#          every_class=every_class, confusion_mat=confusion_mat)
# print("ok")


