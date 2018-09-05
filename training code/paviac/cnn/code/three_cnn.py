from __future__ import division, print_function, absolute_import
import tensorflow as tf
import os
import numpy as np
import cnn_indices
import sys
data = cnn_indices.read_data_sets()
learning_rate = 0.0001
num_steps = 6000
batch_size = 128
display_step = 100

# Network Parameters
# num_input = [17, 17, 200] # MNIST data input (img shape: 28*28)
num_classes = 9  # MNIST total classes (0-9 digits)
# dropout = 0.5 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, 3, 23, 23], name='X')
Y = tf.placeholder(tf.float32, [None, num_classes], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # dropout (keep probability)
# Create some wrappers for simplicity

def conv3d(x, W, b, strides=1):
    x = tf.nn.conv3d(x, W, strides=[1, strides, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool3d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool3d(x, ksize=[1, k, k, k, 1], strides=[1, k, k, k, 1],
                          padding='SAME')
# Create model


def conv_net(x, weights, biases, keep_prob):
    x = tf.reshape(x, [-1, 3, 23, 23, 1])
    conv1 = conv3d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool3d(conv1, k=2)

    # Convolution Layer
    conv2 = conv3d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool3d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, 6*6*1*64])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, keep_prob)
# Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
    # Store layers weight & bias


weights = {
    # c = tf.truncated_normal(shape=[5, 5, 5, 1, 32], mean=0, stddev=1)
    # 'wc1': tf.Variable(tf.truncated_normal(shape=[5, 5, 5, 1, 32], mean=0, stddev=0.01)),
    'wc1': tf.Variable(tf.random_normal([5, 5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([6*6*1*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

logits = conv_net(X, weights, biases, keep_prob)
tf.add_to_collection('pre_prob', logits)
prediction = tf.nn.softmax(logits)
tf.add_to_collection('cnn_pred_label', prediction)
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=Y, logits=logits))
tf.summary.scalar('loss_op', loss_op)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# train_op = optimizer.minimize(loss_op)
train_op = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(loss_op)
# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('batch_accuracy', accuracy)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=50)


with tf.Session() as sess:
    best = 0.7
    sess.run(init)
    merged = tf.summary.merge_all()
    train_summary_writer = tf.summary.FileWriter(
                 '/home/asdf/Documents/juyan/paper/paviac/cnn/model/loss', sess.graph)
    # valid_summary_writer = tf.summary.FileWriter(
    #             '/home/asdf/Documents/juyan/paper/co_training_code/newcode/co_valid_board/cnn')
    for step in range(1, num_steps+1):
        batch_x, batch_y = data.train.next_batch(batch_size)
        batch_x = batch_x.transpose((0, 3, 1, 2))
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.7})
        if step % display_step == 0 or step == 1:
            summary, loss, acc = sess.run([merged, loss_op, accuracy], feed_dict={X: batch_x,
                                                             Y: batch_y, keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))
            # print(logitss)
            # print(pree)
            train_summary_writer.add_summary(summary, step)
        if step % 1000 == 0:
            batch_sizeall = data.valid.num_examples
            val_batch_x, val_batch_y = data.valid.next_batch(batch_sizeall)
            val_batch_x = val_batch_x.transpose((0, 3, 1, 2))
            val_acc, pre = sess.run([accuracy, prediction],
                                             feed_dict={X: val_batch_x, Y: val_batch_y, keep_prob: 1.0})
            # valid_summary_writer.add_summary(summary, step)
            print("valid accuracy = " + "{:.3f}".format(val_acc))
            if val_acc > best:
                best = val_acc
                print("Step " + str(step))
                filename = ('CNN0511.ckpt')
                filename = os.path.join('/home/asdf/Documents/juyan/paper/paviac/cnn/model/',
                                        filename)
                saver.save(sess, filename)
                # filename = ('CNN0421' + str(name_index) + '.ckpt')
                # filename = os.path.join('/home/asdf/Documents/juyan/paper/no_kmeans/model/'
                #                         ''+str(name_index) + '_cnn', filename)
                # saver.save(sess, filename)
            print("best valid accuracy = " + "{:.3f}".format(best))
    print("Optimization Finished!")

