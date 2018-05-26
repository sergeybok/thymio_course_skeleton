#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import cv2
import numpy as np
import re


tf.logging.set_verbosity(tf.logging.INFO)


def create_training_data(folder):
    data_set = []
    labels = []
    for root, directories, filenames in os.walk(folder):
        for directory in directories:
            img_stack = []
            angular_stack = []
            counter = 0
            path_dir = os.path.join(root, directory)
            for root2, directories2, filenames2 in os.walk(path_dir):
                for filename in [f for f in filenames2 if f.endswith('.png')]:
                    path = os.path.join(root2, filename)
                    img = cv2.imread(path, 0)
                    m = re.search(
                        '(\d*)_lin(\d|\d.\d)_ang((-)?(\d|\d.\d))\.png', filename)
                    index = int(m.group(1))
                    linear_vel = float(m.group(2))
                    angular_vel = float(m.group(3))
                    if counter == 0:
                        img_stack = 4 * [img]
                        angular_stack = 4 * [angular_vel]
                        #Shape: (4,84,84)
                        np_img_stack = np.array(img_stack)
                        data_set.append(np.reshape(
                            np_img_stack, [84, 84, 4]))
                        labels.append(np.mean(angular_vel))
                        counter += 1
                    else:
                        img_stack.pop(0)
                        img_stack.append(img[:])
                        angular_stack.pop(0)
                        angular_stack.append(angular_vel)
                        np_img_stack = np.array(img_stack)
                        data_set.append(np.reshape(
                            np_img_stack, [84, 84, 4]))
                        labels.append(np.mean(angular_vel))
                        counter += 1
    if len(data_set) != len(labels):
        print('something is fishy')
    return np.array(data_set), np.array(labels)


import os
import cv2
import numpy as np
import re


# Load an color image in grayscale
#/Users/timonwilli/OneDrive/FS2018/robotics/test_data
def create_test_data(folder):
    test_data_set = []
    test_labels = []
    for root, directories, filenames in os.walk(folder):
        for directory in directories:
            # print(directory)
            img_stack = []
            angular_stack = []
            counter = 0
            path_dir = os.path.join(root, directory)
            for root2, directories2, filenames2 in os.walk(path_dir):
                for filename in [f for f in filenames2 if f.endswith('.png')]:
                    path = os.path.join(root2, filename)
                    # print(path)
                    img = cv2.imread(path, 0)
                    # print(filename)
                    m = re.search(
                        '(\d*)_lin(\d|\d.\d)_ang((-)?(\d|\d.\d))\.png', filename)
                    index = int(m.group(1))
                    linear_vel = float(m.group(2))
                    angular_vel = float(m.group(3))
                    # print(angular_vel)
                    if counter == 0:
                        img_stack = 4 * [img]
                        angular_stack = 4 * [angular_vel]
                    #Shape: (4,84,84)
                        np_img_stack = np.array(img_stack)
                        test_data_set.append(np.reshape(
                            np_img_stack, [84, 84, 4]))
                        test_labels.append(np.mean(angular_vel))
                        counter += 1
                    else:
                        img_stack.pop(0)
                        img_stack.append(img[:])
                        angular_stack.pop(0)
                        angular_stack.append(angular_vel)
                        np_img_stack = np.array(img_stack)
                        test_data_set.append(np.reshape(
                            np_img_stack, [84, 84, 4]))
                        test_labels.append(np.mean(angular_vel))
                        counter += 1
    if len(test_data_set) != len(test_labels):
        print('something is fishy')

    return np.array(test_data_set), np.array(test_labels)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 84, 84, 4])
    print(input_layer.shape)
    scaled_input_layer = tf.to_float(input_layer) / 255.0
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 84, 84, 4]
    # Output Tensor Shape: [batch_size, 84, 84, 32]
    conv1 = tf.layers.conv2d(
        inputs=scaled_input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 84, 84, 32]
    # Output Tensor Shape: [batch_size, 42, 42, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    print(pool1.shape)
    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 42, 42, 32]
    # Output Tensor Shape: [batch_size, 42, 42, 64]
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 42, 42, 64]
    # Output Tensor Shape: [batch_size, 21, 21, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    print(pool2.shape)
    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 21, 21, 64]
    # Output Tensor Shape: [batch_size, 21 * 21 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 42 * 42 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 21 * 21 * 256]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    output_layer = tf.layers.dense(inputs=dropout, units=1)

    predictions = tf.squeeze(output_layer, 1, name='predict_tensor')

    # predictions = {
    #    # Generate predictions (for PREDICT and EVAL mode)
    #    "classes": tf.argmax(input=logits, axis=1),
    #    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    #    # `logging_hook`.
    #    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    #}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(
        labels=labels, predictions=predictions)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "MSE": tf.metrics.mean_squared_error(
            labels=labels, predictions=predictions)}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images  # Returns np.array
    #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images  # Returns np.array
    #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    train_data, train_labels = create_training_data(
        '/Users/timonwilli/OneDrive/FS2018/robotics/training_data')
    print(train_data.shape, train_labels.shape)
    eval_data, eval_labels = create_test_data(
        '/Users/timonwilli/OneDrive/FS2018/robotics/test_data')
    print(eval_data.shape, eval_labels.shape)
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"predictions": "predict_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
