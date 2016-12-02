#
#   mnist_2nets.py
#       date. 11/28/2016, 12/1
#

import numpy as np
import tensorflow as tf

# Import data
from input_data import DataSet, extract_images, extract_labels

# Full-connected Layer   
class FullConnected(object):
    def __init__(self, input, n_in, n_out, vn=('W', 'b')):
        self.input = input

        weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.05)
        bias_init = tf.constant_initializer(value=0.0)
        W = tf.get_variable(vn[0], [n_in, n_out], initializer=weight_init)
        b = tf.get_variable(vn[1], [n_out], initializer=bias_init)
        self.w = W
        self.b = b
        self.params = [self.w, self.b]
    
    def output(self):
        linarg = tf.matmul(self.input, self.w) + self.b
        self.output = tf.nn.relu(linarg)
        
        return self.output
#

# Read-out Layer
class ReadOutLayer(object):
    def __init__(self, input, n_in, n_out, vn=('W', 'b')):
        self.input = input

        weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.05)
        bias_init = tf.constant_initializer(value=0.0)
        W = tf.get_variable(vn[0], [n_in, n_out], initializer=weight_init)
        b = tf.get_variable(vn[1], [n_out], initializer=bias_init)
        self.w = W
        self.b = b
        self.params = [self.w, self.b]
    
    def output(self):
        linarg = tf.matmul(self.input, self.w) + self.b
        self.output = tf.nn.softmax(linarg)  

        return self.output
#

def read_and_split(dirn='../data', one_hot=True):
    class DataSets(object):
        pass
    data_sets = DataSets()

    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    TRAIN1_SIZE = 30000

    BASE_PATH = dirn + '/'
    train_images = extract_images(BASE_PATH+TRAIN_IMAGES)
    train_labels = extract_labels((BASE_PATH+TRAIN_LABELS), one_hot=one_hot)
    test_images = extract_images(BASE_PATH+TEST_IMAGES)
    test_labels = extract_labels((BASE_PATH+TEST_LABELS), one_hot=one_hot)

    train1_images = train_images[:TRAIN1_SIZE]
    train1_labels = train_labels[:TRAIN1_SIZE]
    train2_images = train_images[TRAIN1_SIZE:]
    train2_labels = train_labels[TRAIN1_SIZE:]

    data_sets.train1 = DataSet(train1_images, train1_labels)
    data_sets.train2 = DataSet(train2_images, train2_labels)
    data_sets.test = DataSet(test_images, test_labels)

    return data_sets
 


# Create the model
def mk_NN_model(scope='mlp'):
    with tf.variable_scope(scope):
        hidden1 = FullConnected(x, 784, 625, vn=('W_hid_1','b_hid_1'))
        h1out = hidden1.output()
    
        hidden2 = FullConnected(h1out, 625, 625, vn=('W_hid_2','b_hid_2'))
        h2out = hidden2.output()
    
        readoutlay = ReadOutLayer(h2out, 625, 10, vn=('W_RO', 'b_RO'))
        y_pred = readoutlay.output()
     
        cross_entropy = -tf.reduce_sum(y_*tf.log(y_pred))
    
        # Regularization terms (weight decay)
        L2_sqr = tf.nn.l2_loss(hidden1.w) + tf.nn.l2_loss(hidden2.w)
        lambda_2 = 0.01

        # the loss and accuracy
        with tf.name_scope('loss'):
            loss = cross_entropy + lambda_2 * L2_sqr
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return y_pred, loss, accuracy
 
#

def test_averaging(predicts, actual):
    '''
      test classification process
        args.:
        predicts    : predictions lists by networks
        actual      : label data of test
    '''
    with tf.name_scope('model_averaging'):
        y_pred_ave = (predicts[0] + predicts[1]) / 2.

        correct_prediction = tf.equal(tf.argmax(y_pred_ave,1), tf.argmax(actual,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy
#


if __name__ == '__main__':
    # Load Dataset
    mnist = read_and_split("../data/", one_hot=True)

    # Variables
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    y_pred1, loss1, accuracy1 = mk_NN_model(scope='mlp1')
    y_pred2, loss2, accuracy2 = mk_NN_model(scope='mlp2')

    # Train
    with tf.name_scope('train1'):
        train_step1 = tf.train.AdagradOptimizer(0.003).minimize(loss1)
    with tf.name_scope('train2'):
        train_step2 = tf.train.AdamOptimizer(0.001).minimize(loss2)
    
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter('/tmp/tflogs'+'/train', sess.graph)
        sess.run(init)
        print('Training...')

        with tf.device('/gpu:0'):
            print('  Network No.1 :')
            for i in range(5001):
                batch_xs, batch_ys = mnist.train1.next_batch(100)
                train_step1.run({x: batch_xs, y_: batch_ys})
                if i % 1000 == 0:
                    accuracy1_i = accuracy1.eval({x: batch_xs, y_: batch_ys})
                    loss1_i = loss1.eval({x: batch_xs, y_: batch_ys})
                    print('  step, loss, accurary = {:>6d}:{:>8.3f},{:>8.3f}'\
                        .format(i, loss1_i, accuracy1_i))
            
        with tf.device('/cpu:0'):
            print('  Network No.2 :')
            for i in range(5001):
                batch_xs, batch_ys = mnist.train2.next_batch(100)
                train_step2.run({x: batch_xs, y_: batch_ys})
                if i % 1000 == 0:
                    accuracy2_i = accuracy2.eval({x: batch_xs, y_: batch_ys})
                    loss2_i = loss2.eval({x: batch_xs, y_: batch_ys})
                    print('  step, loss, accurary = {:>6d}:{:>8.3f},{:>8.3f}'\
                        .format(i, loss2_i, accuracy2_i))

        # Test trained model
        accu_ave = test_averaging([y_pred1, y_pred2], y_)
        averaged = sess.run(accu_ave, 
            feed_dict={x: mnist.test.images, y_: mnist.test.labels})

        print('accuracy1 = {:>8.4f}'.format(accuracy1.eval(
            {x: mnist.test.images, y_: mnist.test.labels})))
        print('accuracy2 = {:>8.4f}'.format(accuracy2.eval(
            {x: mnist.test.images, y_: mnist.test.labels})))
        print('accuracy (model averaged) = {:>8.4f}'.format(averaged))

