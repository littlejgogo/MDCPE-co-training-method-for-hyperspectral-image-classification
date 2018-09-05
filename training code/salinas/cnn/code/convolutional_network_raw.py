""" Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import os
import numpy as np
import cnn_indices
data = cnn_indices.read_data_sets()
learning_rate = 0.0000005
num_steps = 50000
batch_size = 30
display_step = 100
num_classes = 9# MNIST total classes (0-9 digits)


# tf Graph input
X = tf.placeholder(tf.float32, [None, 15, 15, 4], name='X')
Y = tf.placeholder(tf.float32, [None, num_classes], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

def weight_variable(shape):
    inital=tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(inital)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')


# Create model
def conv_net(x, weights, biases, keep_prob):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 15, 15, 4])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)



    # conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # # Max Pooling (down-sampling)
    # conv3 = maxpool2d(conv3, k=2)
    # conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': weight_variable([5, 5, 4, 32]),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': weight_variable([3, 3, 32, 64]),
    # fully connected, 7*7*64 inputs, 1024 outputs
    # 'wc3': weight_variable([4, 4, 64, 128]),
    'wd1': weight_variable([1*1*64, 512]),
    # 1024 inputs, 10 outputs (class prediction)
    'out': weight_variable([512, num_classes])
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    # 'bc3': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([512])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
tf.add_to_collection('pre_prob', logits)
prediction = tf.nn.softmax(logits)
tf.add_to_collection('cnn_pred_label', prediction)
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=Y, logits=logits))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# def count1():
#     total_parameters = 0
#     for variable in tf.trainable_variables():
#         # shape is an array of tf.Dimension
#         shape = variable.get_shape()
#         # print(shape)
#         # print(len(shape))
#         variable_parameters = 1
#         for dim in shape:
#             # print(dim)
#             variable_parameters *= dim.value
#         # print(variable_parameters)
#         total_parameters += variable_parameters
#     return total_parameters
#
#
# all_parameter = count1()
# print(all_parameter)



saver = tf.train.Saver(max_to_keep=30)
# Start training


with tf.Session() as sess:
    best = 0.5
    sess.run(init)

    # saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
    # saver.restore(sess, "/home/asdf/Documents/juyan/paper/co_training_code/newcode/041815eachclass/cnn/pretrain_CNN0418.ckpt")


    for step in range(1, num_steps+1):
        batch_x, batch_y = data.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                                  Y: batch_y, keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))


        if step % 2000 == 0:
            # batch_sizeall = data.valid.num_examples
            val_batch_x, val_batch_y = data.valid.next_batch(520)
            val_acc, pre = sess.run([accuracy, prediction], feed_dict={X: val_batch_x, Y: val_batch_y,
                                                                                        keep_prob: 1.0})

            print("valid accuracy = " + "{:.3f}".format(val_acc))
            if val_acc > best:
                best = val_acc
                print("Step " + str(step))
                filename = ('pretrain_CNN0502.ckpt')
                filename = os.path.join('/home/asdf/Documents/juyan/paper/data/paviaumodel/3dcnnmodel',
                                        filename)
                saver.save(sess, filename)
            print("best valid accuracy = " + "{:.3f}".format(best))
    print("Optimization Finished!")
