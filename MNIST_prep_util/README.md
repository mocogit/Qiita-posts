# MNIST_prep_util

This is MNIST data utility to split 4 blocks, which is used for semi-supervised learning.
This code is so-called "add-on" of MNIST data-loader in TensorFlow package.

1. **mnist_prep_ssl.py**
2. **mnist_tfkeras_ssl.py** - demo code using mnist_prep_ssl.py

mnist_prep_ssl.py support 2 mode.

1. "bin-sampling" - you will get label-balanced data(X_lab).
2. "random-sampling" - you will get data(X_lab) including some level of inbalance.

For small number of labelled data, you should use "bin-sampling" mode.
