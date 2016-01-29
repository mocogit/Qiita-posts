#
#   music_scale_classify.py
#       date. 1/22/2016, 1/28
#
#       modeling elman network (of RNN) by 'Theano'
#       Ref. http://peterroelants.github.io/posts/rnn_implementation_part01/
#       Note. This is the simpest RNN
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import theano
import theano.tensor as T


class simpleRNN(object):
    #   members:  slen  : state length
    #             w_x   : weight of input-->hidden layer
    #             w_rec : weight of recurrnce 
    def __init__(self, slen, nx, nrec, ny):
        self.len = slen
        self.w_h = theano.shared(
            np.asarray(np.random.uniform(-.1, .1, (nx)),
            dtype=theano.config.floatX)
        )
        self.w_rec = theano.shared(
            np.asarray(np.random.uniform(-.1, .1, (nrec)),
            dtype=theano.config.floatX)
        )
        self.w_o = theano.shared(
            np.asarray(np.random.uniform(-1., .1, (ny)),
            dtype=theano.config.floatX)
        )
        self.b_h = theano.shared(
            np.asarray(0., dtype=theano.config.floatX)            
        )
        self.b_o = theano.shared(
            np.asarray(0., dtype=theano.config.floatX)
        )
        self.params = [self.w_h, self.w_rec, self.w_o,
                       self.b_h, self.b_o
        ]
    
    def state_update(self, x_t, s0):
        # this is the network updater for simpleRNN
        def inner_fn(xv, s_tm1, wx, wr, wo, bh, bo):
            s_t = xv * wx + s_tm1 * wr + bh
            y_t = T.nnet.sigmoid(s_t * wo + bo)
            
            return [s_t, y_t]
        
        w_h_vec = self.w_h[0]
        w_rec_vec = self.w_rec[0]
        w_o = self.w_o[0]
        b_h = self.b_h
        b_o = self.b_o
        
        [s_t, y_t], updates = theano.scan(fn=inner_fn,
                        sequences=[x_t],
                        outputs_info=[s0, None],
                        non_sequences=[w_h_vec, w_rec_vec, w_o, b_h, b_o]
        )
        return y_t

class ReadOutLayer(object):
    def __init__(self, input, n_in, n_out):
        self.input = input
        
        w_o_np = 0.05 * (np.random.standard_normal([n_in,n_out]))
        w_o = theano.shared(np.asarray(w_o_np, dtype=theano.config.floatX))
        b_o = theano.shared(
            np.asarray(np.zeros(n_out, dtype=theano.config.floatX))
        )
       
        self.w = w_o
        self.b = b_o
        self.params = [self.w, self.b]
    
    def output(self):
        linarg = T.dot(self.input, self.w) + self.b
        self.output = T.nnet.sigmoid(linarg)  

        return self.output
     

class Optimizer(object):
    def __init__(self, params, learning_rate=0.01):
        self.lr = learning_rate
        self.params = params
       
    def minimize(self, loss):
        params = self.params
        self.gradparams = [T.grad(loss, param) for param in params]

        
class GradientDescentOptimizer(Optimizer):
    def __init__(self, params, learning_rate=0.01):
        super(GradientDescentOptimizer, self).__init__(params, learning_rate)
        
    def minimize(self, loss):
        super(GradientDescentOptimizer, self).minimize(loss)
        updates = [
            (param_i, param_i - self.lr * grad_i)
            for param_i, grad_i in zip(self.params, self.gradparams)
        ]
        
        return updates

class RMSPropOptimizer(Optimizer):
    def __init__(self, params, learning_rate=0.01):
        super(RMSPropOptimizer, self).__init__(params, learning_rate)
        self.running_square_ = [theano.shared(np.zeros_like(p.get_value()))
                                for p in params]
        self.running_avg_ = [theano.shared(np.zeros_like(p.get_value()))
                             for p in params]
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]
        
    def minimize(self, loss, momentum, rescale):
        super(RMSPropOptimizer, self).minimize(loss)
        grads = self.gradparams
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        grad_norm = T.sqrt(grad_norm)
        scaling_num = rescale
        scaling_den = T.maximum(rescale, grad_norm)
        # Magic constants
        combination_coeff = 0.9
        minimum_grad = 1E-4
        updates = []
        params = self.params
        for n, (param, grad) in enumerate(zip(params, grads)):
            grad = T.switch(not_finite, 0.1 * param,
                            grad * (scaling_num / scaling_den))
            old_square = self.running_square_[n]
            new_square = combination_coeff * old_square + (
                1. - combination_coeff) * T.sqr(grad)
            old_avg = self.running_avg_[n]
            new_avg = combination_coeff * old_avg + (
                1. - combination_coeff) * grad
            rms_grad = T.sqrt(new_square - new_avg ** 2)
            rms_grad = T.maximum(rms_grad, minimum_grad)
            memory = self.memory_[n]
            update = momentum * memory - self.lr * grad / rms_grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * self.lr * grad / rms_grad
            updates.append((old_square, new_square))
            updates.append((old_avg, new_avg))
            updates.append((memory, update))
            updates.append((param, param + update2))
        
        return updates

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
    
    trX = trX.T       # need 'List of vector' shape dataset
    # trY :'0' stands for 'major', '1' for 'minor'
    trY = [is_minor_scale(keystr) for keystr in trY]
    trY = np.asarray(trY, dtype=np.float32)
    # s0 is time-zero state 
    s0np = np.zeros((num_samples), dtype=np.float32)

    # Tensor Declaration
    x_t = T.matrix('x_t')
    x = T.matrix('x')
    y_ = T.vector('y_')
    s0 = T.vector('s0')
    y_hypo = T.vector('y_hypo')

    # net = simpleRNN(seq_len, seq_len, 1)
    net = simpleRNN(seq_len, 1, 1, 1)

    y_t = net.state_update(x_t, s0)
    y_tt = T.transpose(y_t)
    ro_layer = ReadOutLayer(input=y_tt, n_in=seq_len, n_out=1)
    
    y_hypo = (ro_layer.output()).flatten()
    prediction = y_hypo > 0.5
    
    cross_entropy = T.nnet.binary_crossentropy(y_hypo, y_)
    L2_sqr = (net.w_h ** 2).sum() + (net.w_rec ** 2).sum() \
             + (net.w_o ** 2).sum() + (ro_layer.w **2).sum()
    loss = cross_entropy.mean() + 1.e-04 * L2_sqr    # regularization 

    
    x_t_shape = T.shape(x_t)
    y_t_shape = T.shape(y_t)
    y_hypo_shape = T.shape(y_hypo)
    w_ro_shape = T.shape(ro_layer.w)
    y_lab_shape = T.shape(y_)

    
    # Train Net Model
    params = net.params + ro_layer.params
    # optimizer = GradientDescentOptimizer(params, learning_rate=0.01)
    # train_op = optimizer.minimize(loss)
    optimizer = RMSPropOptimizer(params, learning_rate=0.01)
    train_op = optimizer.minimize(loss, momentum=0.03, rescale=5.)

    # Compile ... define theano.function()
    train_model = theano.function(
        inputs=[],
        outputs=[loss],
        updates=train_op,
        givens=[(x_t, trX), (y_, trY), (s0, s0np)],
        allow_input_downcast=True
    )
    predict = theano.function(
        inputs=[],
        outputs=prediction,
        givens=[(x_t, trX), (s0, s0np)],
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

    
    fp = open("rnn1_log.txt", "w")
    num = len(loss_rec)
    for i in range(num):
        el = str(loss_rec[i]) + ', ' + str(accu_rec[i]) + '\n'
        fp.write(el)
    fp.close()
    
