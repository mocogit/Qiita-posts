#
#   fashion_mnist_preview.py
#       date. 10/12/2017
#

import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

from mnist_imbalanced import prep_imbalanced_dataset

def mk_plot(image_30):
    """
      plot and save image of MNIST data
    """
    n_img = 10      # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n_img):
        # display class-1
        ax = plt.subplot(3, n_img, i + 1)
        plt.imshow(image_30[0, i, :].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display class-2
        ax = plt.subplot(3, n_img, i + 1 + n_img)
        plt.imshow(image_30[1, i, :].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display class-3
        ax = plt.subplot(3, n_img, i + 1 + n_img * 2)
        plt.imshow(image_30[2, i, :].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.8, 
                        hspace=0.2, wspace=0.1)
    plt.savefig('../data/mnist_preview.png')

    return None


if __name__ == '__main__':
    # Load Data
    np.random.seed(seed=201709)
    labels = [1, 3, 5]
    """
      Label	Description
        0	T-shirt/top
        1	Trouser
        2	Pullover
        3	Dress
        4	Coat
        5	Sandal
        6	Shirt
        7	Sneaker
        8	Bag
        9	Ankle boot
    """
    mnist3 = prep_imbalanced_dataset(dirn='../data')
    n_feed = 1000
    samples, labs = mnist3.train.next_batch(n_feed)
    samples_arranged = np.zeros([3, 10, 784], dtype=np.float32)
    n_cnt_by_lab = [0, 0, 0]
    
    # get samples by label [1, 3, 5]
    for i in range(n_feed):
        lab_i = np.argmax(labs[i])
        idx = labels.index(lab_i)

        if n_cnt_by_lab[idx] < 10:
            samples_arranged[idx, n_cnt_by_lab[idx], :] = samples[i, :]
            n_cnt_by_lab[idx] += 1

    mk_plot(samples_arranged)
