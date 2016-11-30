#
#   tf_get_var.py
#       date. 11/30/2016
#

import numpy as np
import tensorflow as tf

b_init = tf.constant_initializer(2.)
b = tf.get_variable('b', shape=[1], initializer=b_init)
c_init = tf.constant_initializer(3.)
c1 = tf.get_variable('c', shape=[1], initializer=c_init)

d = b + c1

init_op = tf.initialize_all_variables()
# init_op = tf.global_variables_initializer()  from v0.12.0
sess = tf.InteractiveSession()

sess.run(init_op)
ds = sess.run(d)
print('d = ', ds)

# This statement occurs error!
# c2 = tf.get_variable('c', shape=[1], initializer=c_init)

with tf.variable_scope('my_vs1'):
    c2 = tf.get_variable('c', shape=[1], initializer=c_init)

c2.initializer.run()

print('c2 = ', sess.run(c2))
print('c1.name = "{}"'.format(c1.name))
print('c2.name = "{}"'.format(c2.name))

#
# How to reuse variable? ... need 'reuse' option
#
e_init = tf.constant_initializer(12345.)

with tf.variable_scope('my_vs2'):
    e1 = tf.get_variable('e', [1], initializer=e_init)

e1.initializer.run()

with tf.variable_scope('my_vs2', reuse=True):
    e2 = tf.get_variable('e', [1])

print('e1 = ', sess.run(e1))
print('e2 = ', sess.run(e2))

# This statement occurs error!
# with tf.variable_scope('my_vs2', reuse=True):
#    e3 = tf.get_variable('e3', [1], initializer=e_init)
