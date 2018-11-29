#!/usr/bin/env python

import tensorflow as tf
import numpy as np

from tensorflow.python.platform import gfile

# Fix for ros kinetic users
import sys
print(sys.path)
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

# OpenCV
import cv2

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Read image
img = cv2.imread('fashion_0.png', cv2.IMREAD_GRAYSCALE)
print('img.shape = ', img.shape)
img = img.astype('float32')
img /= 255.0
img = img.reshape(1, 28, 28, 1)

# Initialize a tensorflow session
with tf.Session() as sess:
    # Load the protobuf graph
    with gfile.FastGFile("models/fashion_mnist.h5.pb",'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Add the graph to the session
        tf.import_graph_def(graph_def, name='')

    # Get graph
    graph = tf.get_default_graph()

    # Get tensor from graph
    pred = graph.get_tensor_by_name("output_class/Softmax:0")

    # Run the session, evaluating our "c" operation from the graph
    res = sess.run(pred, feed_dict={'input_image_input:0': img})

    # Print test accuracy
    pred_index = np.argmax(res[0])

    # Print test accuracy
    print('Predict:', pred_index, ' Label:', class_names[pred_index])

