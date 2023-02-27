#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:38:28 2023

@author: reubenbouaziz
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the Caltech 101 dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.caltech101.load_data()

# Preprocess the data
train_images = train_images.reshape((3060, 300, 300, 3)) / 255.0
test_images = test_images.reshape((1530, 300, 300, 3)) / 255.0
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

# Define the model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(101, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)