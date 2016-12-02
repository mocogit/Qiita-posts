#
# -*- coding: utf-8 -*-
#
#   name_scope_ex.py
#       date. 11/27/2016, 12/2
#

'''
  tf.name_scope と tf.variable_scope は違う

'''

import numpy as np
import tensorflow as tf

# tf.name_scope
with tf.name_scope("my_scope"):
    v1 = tf.get_variable("var1", [1], dtype=tf.float32)
    v2 = tf.Variable(1, name="var2", dtype=tf.float32)
    a = tf.add(v1, v2)

print(v1.name)  # var1:0
print(v2.name)  # my_scope/var2:0
print(a.name)   # my_scope/Add:0


# tf.variable_scope
with tf.variable_scope("my_scope"):
    v1 = tf.get_variable("var1", [1], dtype=tf.float32)
    v2 = tf.Variable(1, name="var2", dtype=tf.float32)
    a = tf.add(v1, v2)

print(v1.name)  # my_scope/var1:0
print(v2.name)  # my_scope_1/var2:0  ... スコープ名がupdateされている．
print(a.name)   # my_scope_1/Add:0


'''
  tf.variable_scopeでは，特殊な機能をサポートするtf.name_scopeか？
  (スコープ管理は，同一(identical)のように見える．)
'''
print('\n')
with tf.name_scope("my_scope2"):
    v3 = tf.Variable(3, name="var3", dtype=tf.float32)
with tf.variable_scope("my_scope2"):
    v4_init = tf.constant_initializer([4.])
    v4 = tf.get_variable("var4", shape=[1], initializer=v4_init)

print(v3.name)  # my_scope2/var3:0
print(v4.name)  # my_scope2/var4:0


'''
  reuseをTrueにしないで, 同じ名前をtf.get_variableすると
'''

with tf.variable_scope("my_scope2"):
    v5 = tf.get_variable("var4", shape=[1], initializer=v4_init)

'''
ValueError: Variable my_scope2/var4 already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:

  File "name_scope_ex.py", line 47, in <module>
    v4 = tf.get_variable("var4", shape=[1], initializer=v4_init)
'''

with tf.variable_scope("my_scope2"):
    tf.get_variable_scope().reuse_variables()
    v5 = tf.get_variable("var4", shape=[1], initializer=v4_init)
print(v5.name)  # my_scope2/var4:0
