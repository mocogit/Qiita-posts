#
#   mnist_loader.py
#       7/7/2017
#       load MNIST dataset
#

import os
import gzip
import pickle
import numpy as np

NUM_CLASSES = 10

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    '''
    Extract the images into a 4D uint8 numpy array [index, y, x, depth].
      Args:
        f: A file object that can be passed into a gzip reader.
      Returns:
        data: A 4D uint8 numpy array [index, y, x, depth].
      Raises:
        ValueError: If the bytestream does not start with 2051.
    '''
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)

    return data


def dense_to_one_hot(labels_dense, num_classes):
    '''
      Convert class labels from scalars to one-hot vectors.
    '''
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
    '''
    Extract the labels into a 1D uint8 numpy array [index].
      Args:
        f: A file object that can be passed into a gzip reader.
        one_hot: Does one hot encoding for the result.
        num_classes: Number of classes for the one hot encoding.
      Returns:
        labels: a 1D uint8 numpy array.
      Raises:
        ValueError: If the bystream doesn't start with 2049.
    '''
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels, num_classes)

    return labels


def load_data(data_dir, subset='train'):
    '''
      load mnist dataset from specified directory
    '''
    if subset=='train':
        fn1 = 'train-images-idx3-ubyte.gz'
        fn2 = 'train-labels-idx1-ubyte.gz'

        with open(os.path.join(data_dir, fn1), 'rb') as fp1:
            images = extract_images(fp1)

        with open(os.path.join(data_dir, fn2), 'rb') as fp2:
            labels = extract_labels(fp2)
        
        return images, labels
    elif subset=='test':
        fn3 = 't10k-images-idx3-ubyte.gz'
        fn4 = 't10k-labels-idx1-ubyte.gz'

        with open(os.path.join(data_dir, fn3), 'rb') as fp3:
            images = extract_images(fp3)

        with open(os.path.join(data_dir, fn4), 'rb') as fp4:
            labels = extract_labels(fp4)

        return images, labels
    else:
        raise NotImplementedError('subset should be either train or test')


if __name__ == '__main__':
    # test mnist loader
    dirn = '../MNIST_data'
    img_te, lab_te = load_data(dirn, subset='test')
    print('image(test).shape = ', img_te.shape)
    print('label(test).shape = ', lab_te.shape)
