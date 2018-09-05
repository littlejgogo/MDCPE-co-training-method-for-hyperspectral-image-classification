
import tensorflow as tf
import cnn_indices

data = cnn_indices.read_data_sets()
import final_index
import numpy as np
saver = tf.train.import_meta_graph('/home/asdf/Documents/juyan/paper/salinas/cnn/model/NEW/'
                                   'CNN0507.ckpt.meta')
batch_size = data.train._num_examples
prediction = np.zeros((1, 16), dtype=np.int32)
# true_label = np.zeros((1, 16), dtype=np.int32)
# cnnlogits = np.zeros((1, 9), dtype=np.float64)
with tf.Session() as sess:
    saver.restore(sess, '/home/asdf/Documents/juyan/paper/salinas/cnn/model/NEW/'
                                   'CNN0507.ckpt')
    y = sess.graph.get_tensor_by_name('Softmax:0')
    X = sess.graph.get_operation_by_name('X').outputs[0]
    keep_prob = sess.graph.get_operation_by_name('keep_prob').outputs[0]
    proba = sess.graph.get_tensor_by_name('Add_1:0')
    # for index in range((data.all._num_examples // batch_size) + 1):
    batch = data.train.next_batch_test(batch_size)
    pre_pro = sess.run(y, feed_dict={X: batch, keep_prob: 1.0})
    prediction = np.concatenate((prediction, pre_pro), axis=0)
        # true_label = np.concatenate((true_label, Y), axis=0)
        # cnnlogits = np.concatenate((cnnlogits, cnn_logits), axis=0)
predict_label = np.argmax(prediction[1:], 1) + 1
# true_label = np.argmax(true_label[1:], 1) + 1
# true_label = np.argmax(Y, 1) + 1
# every_class, confusion_mat = final_index.test_data_index(true_label, predict_label, 16)
# np.savez('/home/asdf/Documents/juyan/paper/salinas/cnn/test/test_zhibiao0522.npz',
#          every_class=every_class, confusion_mat=confusion_mat)
# print("ok")
np.save("/home/asdf/Documents/juyan/paper/salinas/cnn/test/pre_trainlabel.npy", predict_label)
