# MNIST_prep_util

This is MNIST data utility to split 4 blocks, which is used for semi-supervised learning.
This code is so-called "add-on" of MNIST data-loader in TensorFlow package.

1. **mnist_prep.ssl.py**
2. **mnist_tfkeras_ssl.py** - demo code using mnist_prep_ssl.py

## ToDo
Currently, I've implemented only random-sampling for labelled data. I'm going to add "bin-sampling" for very small labelled data condition.
