#
#   mlp_model_classify.py   
#       date. 1/27/2016
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import theano
import theano.tensor as T

from music_scale_classify import Optimizer
from music_scale_classify import GradientDescentOptimizer
from music_scale_classify import RMSPropOptimizer
       
def load_data(filename='music_data.csv'):
    fp = open(filename, 'r')
    
    list_x = []
    list_y = []
    
    while True:
        line = fp.readline()
        
        if not line:
            break
        
        nums = line.split(',')
        scale_label = nums[-1]
        numsi = map(int, nums[:-1])
        
        list_x.append(numsi)
        list_y.append(scale_label.strip(' \n'))

    fp.close()
    
    train_x = np.asarray(list_x, dtype=np.float32)
    train_y = np.asarray(list_y)   
    
    return train_x, train_y
 
 
def is_minor_scale(argstr):
    scale_names = ['Cmj', 'Gmj', 'Dmj', 'Amj', 'Emj', 
                   'Amn', 'Emn', 'Bmn', 'Fsmn', 'Csmn'
    ]    
    index = scale_names.index(argstr)
    
    if index in [5, 6, 7, 8, 9]:
        is_minor = 1.0
    elif index in [0, 1, 2, 3, 4]:
        is_minor = 0.0
    else:
        is_minor = -1.0
    is_minor = np.float32(is_minor)
    
    return is_minor
 
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

# Hidden Layer
class HiddenLayer(object):
    def __init__(self, input, n_in, n_out):
        self.input = input
    
        w_h = theano.shared(floatX(np.random.standard_normal([n_in, n_out])) 
                             * 0.05) 
        b_h = theano.shared(floatX(np.zeros(n_out)))
   
        self.w = w_h
        self.b = b_h
        self.params = [self.w, self.b]
    
    def output(self):
        linarg = T.dot(self.input, self.w) + self.b
        # self.output = T.nnet.relu(linarg)
        self.output = T.nnet.sigmoid(linarg)
        
        return self.output

# Read-out Layer
class ReadOutLayer(object):
    def __init__(self, input, n_in, n_out):
        self.input = input
        
        w_o = theano.shared(floatX(np.random.standard_normal([n_in,n_out]))
                             * 0.05)
        b_o = theano.shared(floatX(np.zeros(n_out)))
       
        self.w = w_o
        self.b = b_o
        self.params = [self.w, self.b]
    
    def output(self):
        linarg = T.dot(self.input, self.w) + self.b
        self.output = T.nnet.sigmoid(linarg)  

        return self.output

# Accuracy
def calc_accuracy(prediction, labels):
    mypred = (prediction()).flatten()
    ilabels = (labels).astype(int)
    accu = (mypred==ilabels).astype(int)
    accu = accu.sum() *1.0 / ilabels.shape[0]
    
    return accu

 
if __name__ == '__main__':
    np.random.seed(seed=1)

    # Read Dataset 
    trX, trY = load_data(filename='music_data4.csv')
    num_samples = trX.shape[0]
    seq_len = trX.shape[1]
    
    trY = [is_minor_scale(keys) for keys in trY]
    trY = np.asarray(trY, dtype=np.float32)

    x = T.matrix('x')
    y_ = T.vector('y')
    y_hypo = T.vector('y_hypo')
    
    h_layer1 = HiddenLayer(input=x, n_in=seq_len, n_out=40)
    h_layer2 = HiddenLayer(input=h_layer1.output(), n_in=40, n_out=40)
    o_layer = ReadOutLayer(input=h_layer2.output(), n_in=40, n_out=1)
    
    params = h_layer1.params + h_layer2.params + o_layer.params
    
    y_hypo = (o_layer.output()).flatten()
    prediction = y_hypo > 0.5
    
    cross_entropy = T.nnet.binary_crossentropy(y_hypo, y_)
    L2_sqr = ((h_layer1.w **2).sum()
             + (h_layer2.w **2).sum()
             + (o_layer.w **2).sum()
    )
    loss = cross_entropy.mean() + 0.001 * L2_sqr
    
    # Train Net Model
    # optimizer = GradientDescentOptimizer(params, learning_rate=0.01)
    # train_op = optimizer.minimize(loss)
    optimizer = RMSPropOptimizer(params, learning_rate=0.01)
    train_op = optimizer.minimize(loss, momentum=0.03, rescale=5.)
    
    train_model = theano.function(
        inputs=[],
        outputs=[loss],
        updates=train_op,
        givens=[(x, trX), (y_, trY)],
        allow_input_downcast=True
    )
    predict = theano.function(
        inputs=[],
        outputs=prediction,
        givens=[(x, trX)],
        allow_input_downcast=True
    )   

    n_epochs = 50001
    epoch = 0
    
    loss_rec = []
    accu_rec = []
        
    while (epoch < n_epochs):
        epoch += 1
        loss = train_model()
        
        if epoch % 100 == 0:
            loss_rec.append(float(loss[0]))
            accu = calc_accuracy(predict, trY)
            accu_rec.append(accu)
        
        if epoch % 1000 == 0:
            print('epoch[%5d] : cost =%8.4f' % (epoch, loss[0]))
    
    accu = calc_accuracy(predict, trY)
    print('accuracy = %12.4f ' % accu)

    
    fp = open("mlp_log.txt", "w")
    num = len(loss_rec)
    for i in range(num):
        el = str(loss_rec[i]) + ',' + str(accu_rec[i]) +'\n'
        fp.write(el)
    fp.close()
    
