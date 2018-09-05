import tensorflow as tf
import cnn_indices

data = cnn_indices.read_data_sets()
import final_index
import numpy as np
saver = tf.train.import_meta_graph('/home/asdf/Documents/juyan/paper/co_training_code/newcode/'
                                            '041815eachclass/model/7_cnn/pretrain_CNN04197.ckpt.meta')
batch_size = 2000
prediction = np.zeros((1, 16), dtype=np.int32)
true_label = np.zeros((1, 16), dtype=np.int32)
with tf.Session() as sess:
    saver.restore(sess, '/home/asdf/Documents/juyan/paper/co_training_code/newcode/'
                                            '041815eachclass/model/7_cnn/pretrain_CNN04197.ckpt')
    y = sess.graph.get_tensor_by_name('Softmax:0')
    X = sess.graph.get_operation_by_name('X').outputs[0]
    keep_prob = sess.graph.get_operation_by_name('keep_prob').outputs[0]
    for index in range((data.test._num_examples // batch_size) + 1):
        batch, Y = data.test.next_batch_test(batch_size)
        predict = sess.run(y, feed_dict={X: batch, keep_prob: 1.0})
        prediction = np.concatenate((prediction, predict), axis=0)
        true_label = np.concatenate((true_label, Y), axis=0)
# predict_label = np.argmax(prediction[1:], 1) + 1
true_label = np.argmax(true_label[1:], 1) + 1
prediction = prediction[1:]
rnnprob = np.load("/home/asdf/Documents/juyan/paper/co_training_code/newcode/test_final/result/rnnresult.npy")
allprob = [x * y for x, y in zip(prediction, rnnprob)]

predict_label = np.argmax(allprob, 1) + 1

every_class, confusion_mat = final_index.test_data_index(true_label, predict_label, 16)
np.savez('/home/asdf/Documents/juyan/paper/data/salinas/0418_15each_class/zhibiao0424_all.npz',
         every_class=every_class, confusion_mat=confusion_mat)
print("ok")

# zhibiao = np.load('/home/asdf/Documents/juyan/paper/data/salinas/0418_15each_class/zhibiao0421_cnnco.npz')
# every_class = zhibiao['every_class']
# confusion_mat = zhibiao['confusion_mat']
print("ok")
