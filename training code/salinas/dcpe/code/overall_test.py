import numpy as np
import os
import sys
import three_cnn
import recurrent_network
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# # #
def get_session(gpu_fraction=0.3):
    """
    This function is to allocate GPU memory a specific fraction
    Assume that you have 6GB of GPU memory and want to allocate ~2GB
    """

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


KTF.set_session(get_session(0.3))  # using 60% of total GPU Memory
os.system("nvidia-smi")  # Execute the command (a string) in a subshell
#
best_train_cnn = np.zeros((20,), dtype=np.float32)
best_train_rnn = np.zeros((20,), dtype=np.float32)


for name_num in range(20):
    print("update sample step:", name_num + 1)
    execfile('/home/asdf/Documents/juyan/paper/salinas/dcpe_result/code/unlabel_test.py')
    print("training cnn step:", name_num + 1)
    best_train_cnn[name_num] = three_cnn.train_cnn(name_index=name_num+1)
    print("training rnn step:", name_num + 1)
    best_train_rnn[name_num] = recurrent_network.train_rnn(name_index=name_num+1)
np.save("/home/asdf/Documents/juyan/paper/salinas/dcpe_result/model/cnn20.npy", best_train_cnn)
np.save("/home/asdf/Documents/juyan/paper/salinas/dcpe_result/model/rnn20.npy", best_train_rnn)