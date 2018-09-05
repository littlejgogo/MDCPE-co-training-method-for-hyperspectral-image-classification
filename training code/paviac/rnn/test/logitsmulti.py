
import tensorflow as tf
import cnn_indices

data = cnn_indices.read_data_sets()
import final_index
import numpy as np
saver = tf.train.import_meta_graph('/home/asdf/Documents/juyan/paper/salinas/cnn/model/NEW/'
                                   'CNN0507.ckpt.meta')
batch_size = data.valid._num_examples
with tf.Session() as sess:
    saver.restore(sess, '/home/asdf/Documents/juyan/paper/salinas/cnn/model/NEW/'
                                   'CNN0507.ckpt')
    y = sess.graph.get_tensor_by_name('Softmax:0')
    X = sess.graph.get_operation_by_name('X').outputs[0]
    keep_prob = sess.graph.get_operation_by_name('keep_prob').outputs[0]

    batch, Y = data.valid.next_batch_test(batch_size)
    predict_label = sess.run(y, feed_dict={X: batch, keep_prob: 1.0})
predict_label = np.argmax(predict_label, 1) + 1
true_label = np.argmax(Y, 1) + 1
every_class, confusion_mat = final_index.test_data_index(true_label, predict_label, 16)
np.savez('/home/asdf/Documents/juyan/paper/salinas/cnn/test/zhibiao0513.npz',
         every_class=every_class, confusion_mat=confusion_mat)
print("ok")


