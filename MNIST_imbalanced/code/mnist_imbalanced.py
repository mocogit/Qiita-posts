# -*- coding: utf-8 -*-
#
#   mnist_imbalanced.py
#       date. 10/13/207
#   ref. https://stackoverflow.com/questions/35155655/loss-function-for-class-imbalanced-binary-classifier-in-tensor-flow
#

import os
import collections
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

Datasets = collections.namedtuple('Datasets', ['train', 'test'])

def prep_imbalanced_dataset(dirn='../data'):
    """
      prepare imbalanced dataset
        label-1: dominant label
        label-3: fewer label (about 5% of lebal-1)
        label-5: fewer label (about 5% of label-1)
    """
    mnist = input_data.read_data_sets(dirn, one_hot=False)
    mnist3 = Datasets(train=None, test=None)

    for subset in [mnist.train, mnist.test]:
        mnist_lab = subset.labels
        idx1 = (mnist_lab == 1)     # 'Trouser' class in Fashion-MNIST
        idx3 = (mnist_lab == 3)     # 'Dress'   class
        idx5 = (mnist_lab == 5)     # 'Sandal'  class

        small = subset.num_examples // 200     # original ...total // 10
        idx1 = [i for i in range(len(idx1)) if idx1[i]]
        idx3 = [i for i in range(len(idx3)) if idx3[i]]
        idx5 = [i for i in range(len(idx5)) if idx5[i]]

        idx_merged = np.concatenate([idx1, idx3[:small], idx5[:small]])

        X_sub = subset.images[idx_merged]
        y_sub = subset.labels[idx_merged]

        # make one-hot label
        y_oh = []
        for lab in y_sub:
            lab_i = np.zeros([10], dtype=np.float32)
            lab_i[lab] = 1.0
            y_oh.append(lab_i)
        y_sub = np.asarray(y_oh)

        # adjust before re-entering into DataSet object
        X_sub= X_sub * 255.

        mnist_sub = DataSet(X_sub, y_sub, reshape=False)
    
        if subset == mnist.train:
            mnist3 = mnist3._replace(train=mnist_sub)
        if subset == mnist.test:
            mnist3 = mnist3._replace(test=mnist_sub)

    return mnist3


# make my model
def cnn_model(x, keep_prob):
    """
      cnn (convolutional neural network) model
    """
    net = tf.layers.conv2d(x, 32, kernel_size=(5, 5),
                           strides=(1, 1), padding='same')
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), 
                                  strides=(2, 2),
                                  padding='same')
    net = tf.layers.conv2d(net, 64, kernel_size=(5, 5),
                           strides=(1, 1), padding='same')
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), 
                                  strides=(2, 2),
                                  padding='same')
    net = tf.reshape(net, [-1, 7*7*64])
    net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
    net = tf.layers.dropout(net, rate=(1.0 - keep_prob))

    y = tf.layers.dense(net, 10, activation=None)

    return y


def inference(x, y_, keep_prob, weights=1.0):
    """
      inference process
    """
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    y_pred = cnn_model(x_image, keep_prob)

    loss = tf.losses.softmax_cross_entropy(y_, y_pred, weights)
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return loss, accuracy, y_pred


def mk_weight_dict(dataset):
    """
      make weight dictionary

      args:
        dataset: mnist dataset (train dataset)
    """

    n_samples = dataset.num_examples
    labels = np.argmax(dataset.labels, axis=1)
    uniq, cnt = np.unique(labels, return_counts=True)

    # calculate weight array
    wt = [(1. / c) for c in cnt]
    # scaling
    wt_tot = [wt[list(uniq).index(lab)] for lab in labels]
    wscale = 1. * n_samples / sum(wt_tot)
    wt_scaled = [w * wscale for w in wt]

    weight_dict = dict([(u, w) for u, w in zip(uniq, wt_scaled)])

    return weight_dict


def get_weights_in_batch(batch_y, weight_dict):
    """
      get weights from batch_y (label) and weight_dict
    """

    batch_lab = np.argmax(batch_y, axis=1)
    weights = np.asarray([weight_dict[key_i] for key_i in batch_lab])

    return weights


if __name__ == '__main__':
    # Load Data
    np.random.seed(seed=201709)
    mnist3 = prep_imbalanced_dataset('../data')
    wt_dict = mk_weight_dict(mnist3.train)

    # Variables
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    wt = tf.placeholder(tf.float32, [None])     # class weights
    keep_prob = tf.placeholder(tf.float32)  

    # main graph
    loss, accuracy, y_pred = inference(x, y_, keep_prob, wt)
    train_step = tf.train.AdamOptimizer(
                    learning_rate=5.e-3).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        print('Training...')
        for i in range(2001):
            batch_xs, batch_ys = mnist3.train.next_batch(100)
            batch_wt = get_weights_in_batch(batch_ys, wt_dict)
            train_step.run({x: batch_xs, y_: batch_ys, 
                            keep_prob: 0.5, wt: batch_wt})
            if i % 200 == 0:
                train_accuracy = accuracy.eval(
                    {x: batch_xs, y_: batch_ys, keep_prob: 1.0})
                print('  step, accurary = %6d: %8.4f' % (i, train_accuracy))
        # check dataflow status
        print('epoch comp = %4d times' % mnist3.train.epochs_completed)

        # Test trained model
        print('Testing...')
        n_loop = mnist3.test.num_examples // 100
        y_pred_np = []
        y_true_np = []
        for i in range(n_loop):
            batch_xte, batch_yte = mnist3.test.next_batch(100)
            batch_wt = get_weights_in_batch(batch_yte, wt_dict)
            test_fd = {x: batch_xte, y_: batch_yte, keep_prob: 1.0}
            y_pred_np.extend(sess.run(y_pred, feed_dict=test_fd))
            y_true_np.extend(batch_yte)

    # confusion matrix
    y_true_np = np.asarray(np.argmax(y_true_np, axis=1))
    y_pred_np = np.asarray(np.argmax(y_pred_np, axis=1))
    print('accuracy = {:>8.4f}\n'.format(
            accuracy_score(y_true_np, y_pred_np)))
    print('confusion matrix = \n', 
          confusion_matrix(y_true_np, y_pred_np))
