#!/usr/bin/env python

import tensorflow as tf
import numpy as np

import numpy as np
from tensorflow.python.platform import gfile

# Dataset
import utils.mnist_reader as mnist_reader

train_images, train_labels = mnist_reader.load_mnist('data/fashion', kind='train')
test_images, test_labels = mnist_reader.load_mnist('data/fashion', kind='t10k')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("train_images.shape = {}".format(train_images.shape))

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

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
    pred = graph.get_tensor_by_name("output_class/Softmax")
    #
    # Run the session, evaluating our "c" operation from the graph
    res = sess.run(pred, feed_dict={'input_image_input:0': test_images})

    # Print test accuracy
    pred_index = np.argmax(res[0])

    # Print test accuracy
    print('Predict:', pred_index, ' Label:', class_names[pred_index], 'GT:', test_labels[0])

