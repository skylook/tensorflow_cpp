#!/usr/bin/env python

import tensorflow as tf
import numpy as np

import numpy as np
from tensorflow.python.platform import gfile

# Initialize a tensorflow session
with tf.Session() as sess:
    # Load the protobuf graph
    with gfile.FastGFile("model/simple.pb",'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Add the graph to the session
        tf.import_graph_def(graph_def, name='')

    # Get graph
    graph = tf.get_default_graph()

    # Get tensor from graph
    c = graph.get_tensor_by_name("c:0")

    # Run the session, evaluating our "c" operation from the graph
    res = sess.run(c, feed_dict={'a:0': 2.0, 'b:0': 3.0})

    print("res = ", res)


