#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.python.framework.graph_util import convert_variables_to_constants

with tf.Session() as sess:

    a=tf.placeholder(tf.float32,shape=None, name='a')
    b=tf.placeholder(tf.float32,shape=None, name='b')
    c = tf.multiply(a, b, name='c')

    sess.run(tf.global_variables_initializer())

    graph = convert_variables_to_constants(sess, sess.graph_def, ["c"])
    # tf.train.write_graph(graph, '.', 'graph.pb', as_text=False)
    tf.train.write_graph(sess.graph_def, 'model/', 'simple.pb', as_text=False)

    res = sess.run(c, feed_dict={'a:0': 2.0, 'b:0': 3.0})
    print("res = ", res)