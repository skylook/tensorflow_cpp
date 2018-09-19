#!/usr/bin/env python

import tensorflow as tf
import numpy as np

with tf.Session() as sess:

    a=tf.placeholder(tf.float32,shape=None, name='a')
    b=tf.placeholder(tf.float32,shape=None, name='b')
    c = tf.multiply(a, b, name='c')

    sess.run(tf.global_variables_initializer())

    tf.train.write_graph(sess.graph_def, 'model/', 'simple.pb', as_text=False)

    res = sess.run(c, feed_dict={'a:0': 2.0, 'b:0': 3.0})
    print("res = ", res)