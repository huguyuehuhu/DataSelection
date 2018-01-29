# encoding: utf-8

"""
@author: huguyuehuhu
@time: 17-9-22 下午7:18
"""
import tensorflow as tf


import tensorflow as tf
import numpy as np
import pandas as pd
import time


# restrict models GPU memory utilization to min required
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

a= pd.Series(np.array([7,6,3,4,2,3,4,6]))
b=a.rank(method='first',ascending=False)
c=np.array((b))

aaa=1




























# def fun():
#     a = tf.placeholder(tf.float32)
#     b= tf.placeholder(tf.float32)
#     c = a * b
#     d=c+1
# # # Launch the graph in a session.
#     sess = tf.Session()
#     return sess,d,a,b
#
# sess,d,a,b=fun()
# e=1
# f=2
# print(sess.run(d, feed_dict={a: e, b: f}))

#
# class MultiPerceptron:
#     def __init__(self):
#         # Network Parameters
#         self.n_classes = 10  # MNIST total classes (0-9 digits)
#
#         self._define_inputs()
#
#     # tf Graph input
#     def _define_inputs(self):
#         self.a=12
# c=1
# def func2(models):
#     models.a=13
#
# def func1():
#     b = MultiPerceptron()
#     #func2(b)
#     return b
# d=func1()
# f=d.a
# k=1


# import pandas as pd
# from numpy# import tensorflow as tf
# import numpy as np
# import time
#
#
# # restrict models GPU memory utilization to min required
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# sess.run(tf.global_variables_initializer())
#
# import *
# import matplotlib.pyplot as plt
# ts = pd.Series(random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
# ts = ts.cumsum()
# ts.plot()
# # must have this to show
# plt.show()

# import tensorflow as tf
# import numpy as np
# import time
#
#
# # restrict models GPU memory utilization to min required
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# sess.run(tf.global_variables_initializer())
#
# # We define a Variable
# x = tf.Variable(0, dtype=tf.int32)
# k = tf.Variable(2, dtype=tf.int32)
# k1 = tf.Variable(tf.random_normal([10, 100]))
# k2= tf.Variable(2, dtype=tf.int32)
# k3 = tf.Variable(2, dtype=tf.int32)
# k4 = tf.Variable(2, dtype=tf.int32)
#
# # We use a simple assign operation
# assign_op = tf.assign(x, x + 1)
# f = tf.trainable_variables()
# #var = [var for var in f ]
# a= np.random.random(3)
# c_old=np.zeros(2)
#
# d=np.array([])
#
#
# g=[]
# t1 = time.time()
# c1 = sess.run([var for var in f])
# for i in range(1):
#     # print('x:', sess.run(x))
#     # sess.run(assign_op)
#
#     c2 = sess.run([var for var in f])
#     #v = list(map(lambda x: (x[0] - x[1]), zip(c2, c1)))
#     v = list(map(lambda x: (x[0] - x[1]).ravel(), zip(c2, c1)))
#     # for var in v:
#     #     d = np.hstack((d, var))
#
#     # delta_c = c - c_old
#     # c_old = c
#     # print('c:', delta_c)
# print(time. time()-t1)