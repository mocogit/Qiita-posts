#
#   mnist_tfkeras_ssl.py
#       date. 4/27/2017
#       Semi-supervised learning demo

import numpy as np

from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Dense, Input
from tensorflow.contrib.keras.python.keras.layers import Dropout
from tensorflow.contrib.keras.python.keras.optimizers import Adagrad, SGD
from tensorflow.contrib.keras.python.keras.utils import np_utils
from tensorflow.contrib.keras.python.keras import backend as K

from mnist_prep_ssl import load_data_ssl

def params():
    '''
      parameters for data allocation
    '''
    params = {}
    params['n_train_lab'] = 1000
    params['n_val'] = 5000
    # total train samples = 60,000
    params['n_train_unlab'] = 60000 - 1000 - 5000
    params['percent_limit'] = (0.8, 1.2)

    return params

def mlp_encoder():
    input_img = Input(shape=(784,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)

    encoder = Model(input_img, encoded)

    return encoder

def mk_ae_model():
    input_img = Input(shape=(784,))
    encoder = mlp_encoder()
    encoded = encoder(input_img)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)

    autoencoder = Model(input_img, decoded)

    return autoencoder, encoder

def mk_clf_model(encoder):
    input_img = Input(shape=(784,))
    encoded = encoder(input_img)
    
    for layer in encoder.layers:
        layer.trainable = True
        # if False, accuracy is worse.
    # additional layers
    net = Dropout(0.2)(encoded)
    net = Dense(96, activation='relu', trainable=True)(net)
    readout = Dense(10, activation='softmax', trainable=True)(net)

    classifier = Model(input_img, readout)

    return classifier


if __name__ == '__main__':
    np.random.seed(201604)  # for reproducibility
    batch_size = 128

    mnist = load_data_ssl(params(), dirn='../../MNIST_data')

    model_ae, model_encoder = mk_ae_model()
    model_ae.compile(loss='binary_crossentropy',
                     optimizer=Adagrad(lr=0.01))

    # Pre-traininging process
    nloops = int(mnist.train_unlab.num_examples / batch_size)
    epochs_pre = 20

    print('\nTraining Autoencoder ...')
    for e in range(1, epochs_pre+1):
        for i in range(nloops):
            batch_x, _ = mnist.train_unlab.next_batch(batch_size)
            loss_tr = model_ae.train_on_batch(batch_x, batch_x)

            if i % 100 == 0:
                batch_xv, _ = mnist.validation.next_batch(100)
                loss_val = model_ae.test_on_batch(batch_xv, batch_xv)
                # val_acc = model_ae.evaluate(batch_xv, batch_xv, verbose=0)[1]
                print('epoch/batch = {:4d} /{:4d}, loss_tr = {:>.4f}, '
                      'loss_val = {:>.4f}'.format(e, i, loss_tr, loss_val))

    # Finetuning process
    batch_size = 100
    nloops = int(mnist.train_lab.num_examples / batch_size)
    epochs_ft = 100
    model_clf = mk_clf_model(model_encoder)
    model_clf.compile(loss='categorical_crossentropy',
                     optimizer=SGD(lr=0.02),
                     metrics=['accuracy'])

    print('\nTraining Classifier ...')
    for e in range(1, epochs_ft+1):
        for i in range(nloops):
            # use labels(y) in this process
            batch_x, batch_y = mnist.train_lab.next_batch(batch_size)
            loss_tr = model_clf.train_on_batch(batch_x, batch_y)

        batch_xv, batch_yv = mnist.validation.next_batch(100)
        loss_val = model_clf.test_on_batch(batch_xv, batch_yv)

        if e % 10 == 0:
            print('epoch{:4d}, loss(tr/val) = ({:>.4f}, {:>.4f}), '
                  'accuracy(tr/val) = ({:>.4f}, {:>.4f})'.format(
                       e, loss_tr[0], loss_val[0], loss_tr[1], loss_val[1]))

    X_test = mnist.test.images
    y_test = mnist.test.labels
    score = model_clf.evaluate(X_test, y_test, verbose=0)
    print('\nTest accuracy: {:>.4f}'.format(score[1]))

    K.clear_session()
