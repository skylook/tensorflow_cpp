# TensorFlow and tf.keras
import os
import gzip
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

from tensorflow.keras import backend as K

# Use this only for export of the model.
K.set_learning_phase(0)
K.set_image_data_format('channels_last')
sess = K.get_session()

# Helper libraries
import numpy as np

# Plot model
from tensorflow.keras.utils import plot_model

# Dataset
import utils.mnist_reader as mnist_reader

# Model
from model import create_model

print(tf.__version__)
print(keras.__version__)

train_images, train_labels = mnist_reader.load_mnist('data/fashion', kind='train')
test_images, test_labels = mnist_reader.load_mnist('data/fashion', kind='t10k')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

# convert class vectors to binary class matrices
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Create CNN Model from model.py
model = create_model()

# Take a look at the model summary
model.summary()

# Virtualize model
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png')

# Include the epoch in the file name. (uses `str.format`)
checkpoint_path="train_logs/cp-{epoch:04d}.hdf5"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=False,
    # Save weights, every 5-epochs.
    period=5)

# Comile model with loss and optimizer
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model
model.fit(train_images,
          train_labels,
          batch_size=64,
          callbacks = [cp_callback],
          epochs=10)

model.evaluate(test_images, test_labels)

# Evaluate the model on test set
score = model.evaluate(test_images, test_labels, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# Predict using Keras
predictions = model.predict(test_images)
pred_index = np.argmax(predictions[0])

# Print test accuracy
print('Predict:', pred_index, ' Label:', class_names[pred_index], 'GT:', test_labels[0])

# Save whole graph & weights
model_path = "models/fashion_mnist.h5"
model.save(model_path)

print('Finish writing model to : {}'.format(model_path))
print('You can convert model to tensorflow format:\npython3 utils/keras_to_tensorflow.py -input_model_file {} -output_model_file {}'.format(model_path, model_path + ".pb"))