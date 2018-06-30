import math
import ast
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
import pydicom
from tensorflow.python.framework import ops
from cnn_utils import *

def create_placeholders(n_H0=256, n_W0=256, n_C0=1, n_y=3):
    X = tf.placeholder(tf.float32, (None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, (None, n_y))
    return X, Y


def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [3, 3, 1, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W3 = tf.get_variable("W3", [3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W4 = tf.get_variable("W4", [3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters = {"W1": W1, "W2": W2, "W3": W3, "W4": W4}
    return parameters


def forward_propagation(X, parameters):
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']

### START CODE HERE ###
# CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
# RELU
    A1 = tf.nn.relu(Z1)
# MAXPOOL: window 2x2, sride 2, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
# CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
# RELU
    A2 = tf.nn.relu(Z2)
# MAXPOOL: window 2x2, stride 2, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
# CONV2D: stride of 1, padding 'SAME'
    Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='SAME')
# RELU
    A3 = tf.nn.relu(Z3)
# MAXPOOL: window 2x2, sride 2, padding 'SAME'
    P3 = tf.nn.max_pool(A3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='VALID')  # CONV2D: filters W2, stride 1, padding 'SAME'
    Z4 = tf.nn.conv2d(P3, W4, strides=[1, 1, 1, 1], padding='SAME')
# RELU
    A4 = tf.nn.relu(Z4)
# MAXPOOL: window 2x2, stride 2, padding 'SAME'
    P4 = tf.nn.max_pool(A4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
# FLATTEN
    P4 = tf.contrib.layers.flatten(P4)
# FULLY-CONNECTED without non-linear activation function (not not call softmax).
# 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z5 = tf.contrib.layers.fully_connected(P4, 4096, activation_fn=tf.nn.relu)
    Z6 = tf.contrib.layers.fully_connected(Z5, 3, activation_fn=None)
### END CODE HERE ###
    return Z6


def compute_cost(Z6, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z6, labels=Y))
    return cost


def model(learning_rate=0.009, num_epochs=10, print_cost=True):
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables

    df_img_path = pd.read_csv('./pixel2.csv',index_col=False)
    df_train_complete=df_img_path.loc[0:21632]
    df_valid1=df_img_path.loc[0:3000]
    df_train1=df_img_path.loc[3001:21632]
    df_valid2 = df_img_path.loc[3001:6000]
    df_train2 = df_img_path.loc[0:3000]
    df_train2 = df_train2.append(df_img_path.loc[6001:21362])
    df_valid3 = df_img_path.loc[6001:9000]
    df_train3 = df_img_path.loc[0:6000]
    df_train3 = df_train3.append(df_img_path.loc[9001:21632])
    df_valid4 = df_img_path.loc[9001:12000]
    df_train4 = df_img_path.loc[0:9000]
    df_train4 = df_train4.append(df_img_path.loc[12001:21362])
    df_valid5 = df_img_path.loc[12001:15000]
    df_train5 = df_img_path.loc[0:12000]
    df_train5 = df_train5.append(df_img_path.loc[15001:21362])
    df_valid6 = df_img_path.loc[15001:18000]
    df_train6 = df_img_path.loc[0:15000]
    df_train6 = df_train5.append(df_img_path.loc[18001:21362])
    df_valid7 = df_img_path.loc[18001:21000]
    df_train7 = df_img_path.loc[0:18000]
    df_train7 = df_train5.append(df_img_path.loc[21001:21362])

    df_test= df_img_path.loc[50000:70964]
    tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    seed = 3  # to keep results consistent (numpy seed)
    costs = []  # To keep track of the cost
    saver = tf.train.Saver()
    X, Y = create_placeholders()
    parameters = initialize_parameters()
    Z6 = forward_propagation(X, parameters)
    cost = compute_cost(Z6, Y)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    chunksize = 512
    num_batches_train = 120
    num_batches_test = 7
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            mcost = 0  # minibatch cost
            for df in pd.read_csv('training.csv', chunksize=chunksize, header=None, iterator=True):
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: ast.literal_eval(df.iloc[:, 0]), Y: df.iloc[:, 1]})
                mcost += temp_cost / 120

        # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, mcost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(mcost)

    # plot the cost
# plt.plot(np.squeeze(costs))
# plt.ylabel('cost')
# plt.xlabel('iterations (per tens)')
# plt.title("Learning rate =" + str(learning_rate))
# plt.show()

# Calculate the correct predictions
        predict_op = tf.argmax(Z6, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

# Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = 0
        test_accuracy = 0
        for df in pd.read_csv('training.csv', chunksize=chunksize, header=None, iterator=True):
            train_accuracy += accuracy.eval({X: ast.literal_eval(df.iloc[:, 0]), Y: df.iloc[:, 1]})
        for test in pd.read_csv('testing.csv', chunksize=chunksize, header=None, iterator=True):
            test_accuracy += accuracy.eval({X: ast.literal_eval(test.iloc[:, 0]), Y: test.iloc[:, 1]})
        print("Train Accuracy:", train_accuracy / 120)
        print("Test Accuracy:", test_accuracy / 7)
        #save_path = saver.save(sess, "./model1.ckpt")

        return train_accuracy, test_accuracy, parameters

_, _, parameters = model()
