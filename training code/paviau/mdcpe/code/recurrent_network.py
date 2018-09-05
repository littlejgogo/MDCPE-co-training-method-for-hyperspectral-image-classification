""" Recurrent Neural Network.

A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function
import os

from tensorflow.contrib import rnn
import tensorflow as tf
import numpy as np
import keras.backend.tensorflow_backend as KTF
import sys
import rnn_indices
data = rnn_indices.read_data_sets()

# Training Parameters
g_rnn = tf.Graph()
with g_rnn.as_default():
    learning_rate = 0.001
    training_steps = 20000
    batch_size = 128
    display_step = 100

    # Network Parameters
    num_input = 17  # MNIST data input (img shape: 28*28)
    timesteps = 6  # timesteps
    num_hidden = 120  # hidden layer num of features
    num_classes = 9  # MNIST total classes (0-9 digits)

    X = tf.placeholder("float", [None, timesteps, num_input], name='X')
    Y = tf.placeholder("float", [None, num_classes], name='Y')

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
    }
    biases = {
        'out': tf.constant(0.1, shape=([num_classes, ]))
    }


    def RNN(x, weights, biases):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        # x = tf.unstack(x, timesteps, 1)
        # Define a lstm cell with tensorflow
        gru_cell = rnn.GRUCell(num_hidden)
        # init_state = gru_cell.zero_state(dtype=tf.float32)
        # Get lstm cell output
        outputs, states = tf.nn.dynamic_rnn(gru_cell, x, dtype=tf.float32)
        # outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=init_state, time_major=False, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
        results = tf.matmul(outputs[-1], weights['out']) + biases['out']
        return results
        # return tf.matmul(states[1], weights['out']) + biases['out']

    logits = RNN(X, weights, biases)
    tf.add_to_collection('pre_prob', logits)
    prediction = tf.nn.softmax(logits)
    tf.add_to_collection('rnn_pred_label', prediction)
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    # tf.summary.scalar('loss_op', loss_op)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # tf.summary.scalar('batch_accuracy', accuracy)
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training

    saver = tf.train.Saver(max_to_keep=50)


def train_rnn(name_index):
    with tf.Session(graph=g_rnn) as sess:
        best = 0.7
        # Run the initializer
        sess.run(init)
        # saver.restore(sess, "/home/asdf/Documents/juyan/paper/co_training_code/newcode/model/pretrain_RNN.ckpt")
        #
        # merged = tf.summary.merge_all()
        # train_summary_writer = tf.summary.FileWriter(
        #     '/home/asdf/Documents/juyan/paper/co_training_code/newcode/train/rnn/loss_record', sess.graph)
        # valid_summary_writer = tf.summary.FileWriter(
        #     '/home/asdf/Documents/juyan/paper/co_training_code/newcode/valid/rnn_loss_record')
        for step in range(1, training_steps+1):
            batch_x, batch_y = data.train.next_batch(batch_size)
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
                # train_summary_writer.add_summary(summary, step)
            #start validation
            if step % 1000 == 0:
                batch_sizeall = data.valid.num_examples
                val_batch_x, val_batch_y = data.valid.next_batch(batch_sizeall)
                val_batch_x = val_batch_x.reshape((val_batch_x.shape[0], timesteps, num_input))
                val_acc = sess.run(accuracy, feed_dict={X: val_batch_x, Y: val_batch_y})
                # valid_summary_writer.add_summary(summary, step)
                print("valid accuracy = " + "{:.3f}".format(val_acc))
                if val_acc > best:
                    best = val_acc
                    print("Step " + str(step))
                    filename = ('RNN0509.ckpt')
                    filename = os.path.join('/home/asdf/Documents/juyan/paper/paviau/mdcpe/newmdcpe/model/rnn',
                                            filename)
                    saver.save(sess, filename)
                    filename = ('RNN0509'+str(name_index)+'.ckpt')
                    filename = os.path.join('/home/asdf/Documents/juyan/paper/paviau/mdcpe/newmdcpe/model/'
                                            + str(name_index) + '_rnn', filename)
                    saver.save(sess, filename)
                print("best valid accuracy = " + "{:.3f}".format(best))
        print("Optimization Finished!")
    return best

