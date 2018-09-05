from __future__ import division, print_function, absolute_import
import tensorflow as tf
import os
import numpy as np
import cnn_indices
# import keras.backend.tensorflow_backend as KTF
# def get_session(gpu_fraction=0.3):
#     """
#     This function is to allocate GPU memory a specific fraction
#     Assume that you have 6GB of GPU memory and want to allocate ~2GB
#     """
#
#     num_threads = os.environ.get('OMP_NUM_THREADS')
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#
#     if num_threads:
#         return tf.Session(config=tf.ConfigProto(
#             gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
#     else:
#         return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#
#
# KTF.set_session(get_session(0.5))  # using 60% of total GPU Memory
# os.system("nvidia-smi")  # Execute the command (a string) in a subshell
# raw_input("Press Enter to continue...")

data = cnn_indices.read_data_sets()
learning_rate = 0.000001
num_steps = 60000
batch_size = 35
display_step = 100

# Network Parameters
# num_input = [17, 17, 200] # MNIST data input (img shape: 28*28)
num_classes = 9 # MNIST total classes (0-9 digits)
# dropout = 0.5 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, 4, 15, 15], name='X')
Y = tf.placeholder(tf.float32, [None, num_classes], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')# dropout (keep probability)


# Create some wrappers for simplicity
def conv3d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation

    x = tf.nn.conv3d(x, W, strides=[1, strides, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool3d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool3d(x, ksize=[1, k, k, k, 1], strides=[1, k, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, keep_prob):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    # x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    x = tf.reshape(x, [-1, 4, 15, 15, 1])
    conv1 = conv3d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool3d(conv1, k=2)

    # Convolution Layer
    conv2 = conv3d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool3d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, 4*4*1*64])
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
    'wd1': tf.Variable(tf.random_normal([4*4*1*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
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
tf.summary.scalar('loss_op', loss_op)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('batch_accuracy', accuracy)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=30)
# Start training


with tf.Session() as sess:
    best = 0.5
    sess.run(init)

    # saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
    # saver.restore(sess, "/home/asdf/Documents/juyan/paper/co_training_code/newcode/041815eachclass/cnn/pretrain_CNN0418.ckpt")

    merged = tf.summary.merge_all()
    # writer_train = tf.summary.FileWriter("/home/asdf/Documents/juyan/paper/data/paviaumodel/3dcnnmodel/loss")
    # writer_valid = tf.summary.FileWriter("/home/asdf/Documents/juyan/paper/data/paviaumodel/3dcnnmodel/loss")
    train_summary_writer = tf.summary.FileWriter(
                '/home/asdf/Documents/juyan/paper/data/paviaumodel/3dcnnmodel/loss/train', sess.graph)
    valid_summary_writer = tf.summary.FileWriter(
                '/home/asdf/Documents/juyan/paper/data/paviaumodel/3dcnnmodel/loss/valid')
    for step in range(1, num_steps+1):
        batch_x, batch_y = data.train.next_batch(batch_size)
        batch_x = batch_x.transpose((0, 3, 1, 2))
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.6})
        if step % display_step == 0 or step == 1:
            summary, loss, acc = sess.run([merged, loss_op, accuracy], feed_dict={X: batch_x,
                                                                                  Y: batch_y, keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))
            train_summary_writer.add_summary(summary, step)

        if step % 2000 == 0:
            batch_sizeall = data.valid.num_examples
            val_batch_x, val_batch_y = data.valid.next_batch(batch_sizeall)
            val_batch_x = val_batch_x.transpose((0, 3, 1, 2))
            summary, val_acc, pre = sess.run([merged, accuracy, prediction], feed_dict={X: val_batch_x, Y: val_batch_y,
                                                                                        keep_prob: 1.0})
            valid_summary_writer.add_summary(summary, step)
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
