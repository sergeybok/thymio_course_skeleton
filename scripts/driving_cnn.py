#!/usr/bin/env python

import numpy as np
import tensorflow as tf

import roslib;
import rospy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import numpy as np
import cv2, cv_bridge

import sys, select, termios, tty, os

STACKS = 4
CHANNELS = 1

def vels(speed,turn):
	return "currently:\tspeed %s\tturn %s " % (speed,turn)

def cnn_model_fn(features, labels, mode):
        """Model function for CNN."""

        input_layer = tf.reshape(features["x"], [-1, 84, 84, 4])
        print(input_layer.shape)
        scaled_input_layer = tf.to_float(input_layer) / 255.0

        conv1 = tf.layers.conv2d(
            inputs=scaled_input_layer,
            filters=20,
            kernel_size=[5, 5],
            padding="valid",
            activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        print(pool1.shape)

        conv2_out = tf.layers.conv2d(
            inputs=pool1,
            filters=20,
            kernel_size=[5, 5],
            padding="valid",
            activation=tf.nn.relu)
        print(conv2_out.shape)

        pool2 = tf.layers.max_pooling2d(
            inputs=conv2_out, pool_size=[6, 6], strides=6)
        print(pool2.shape)
        pool2_flat = tf.reshape(pool2, [-1, 6 * 6 * 20])


        output_layer = tf.layers.dense(
            inputs=pool2_flat, units=1, activation=tf.nn.tanh)

        predictions = tf.squeeze(output_layer, 1, name='predict_tensor')

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = tf.losses.mean_squared_error(
            labels=labels, predictions=predictions)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        eval_metric_ops = {
            "MSE": tf.metrics.mean_squared_error(
                labels=labels, predictions=predictions)}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

class BasicThymio:

    def __init__(self, thymio_name):
        """init"""
        self.thymio_name = thymio_name
        rospy.init_node('basic_thymio_controller', anonymous=True)

        # Publish to the topic '/thymioX/cmd_vel'.
        # self.image_subscriber = rospy.Subscriber('/{0}/camera/image_raw'.format(thymio_name), Image, callback_image_save)
	self.pub = rospy.Publisher('/{0}/cmd_vel'.format(thymio_name), Twist, queue_size = 1)

        # A subscriber to the topic '/turtle1/pose'. self.update_pose is called
        # when a message of type Pose is received.
        self.image = np.zeros((1,84,84,STACKS))
        self.image_buffer = []
        self.dataset = []
        self.current_twist = Twist()
        # publish at this rate
        self.rate = rospy.Rate(10)



   # def callback_image_save(data):
   #     try:
    #        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
   #     except cv_bridge.cvbridgeerror as e:
   #         print(e)

    #    (rows,cols,channels) = cv_image.shape
   #     if cols > 60 and rows > 60 :
   #         cv2.circle(cv_image, (50,50), 10, 255)
   #     try:
   #         cv_image = cv2.resize(cv_image,(84,84))
   #         cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
   #     except:
   #         print('Grayscale or Resizzle')
   #     if len(self.image_buffer) == STACKED + 1:
   #         del self.image_buffer[0]
   #         c = CHANNELS * STACKED
   #         dx = np.zeros((cv_image.shape[0], cv_image.shape[1], c))
   #         for i in range(STACKED):
   #             c = i * CHANNELS
   #             dx[:, :, c:c + CHANNELS] = np.array(buffer[i])
   #         self.image = dx

    def move(self):
        if len(sys.argv) == 2:
            thymio_name = sys.argv[1]
        else:
            print('Specify thymio name, defaulting to thymio10')
            thymio_name = 'thymio10'

	pub = rospy.Publisher('/{0}/cmd_vel'.format(thymio_name), Twist, queue_size = 1)
        # sensor_center_subscriber = rospy.Subscriber('/{0}/camera/image_raw'.format(thymio_name), Image, callback_image_save)

	speed = rospy.get_param("~speed", 0.2)
	turn = rospy.get_param("~turn", 1.0)
	x = 0
	y = 0
	z = 0
	th = 0
        lin = 0.2
	status = 0
        mnist_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn, model_dir="/home/timonwilli/catkin_ws/src/thymio_course_skeleton/scripts/model_cnn")

	try:
                twist = Twist()
                lin_speed = 0.2
                print(vels(speed,turn))

                while(1):
                    simple_image = self.image
                    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": simple_image}, batch_size=1, shuffle=False)
                    prediction = mnist_classifier.predict(input_fn=train_input_fn)
                    pred = list(prediction)
                    ang = pred[0]
                    global_lin = lin
                    global_ang = ang
                    twist.linear.x = lin; twist.linear.y = 0; twist.linear.z = 0;
                    twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = ang
                    pub.publish(twist)
        except Exception as e:
            print(e)

	finally:
		twist = Twist()
		twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
		twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
		pub.publish(twist)

def usage():
        return "Wrong number of parameters. basic_move.py [thymio_name]"

if __name__ == '__main__':
    if len(sys.argv) == 2:
        thymio_name = sys.argv[1]
        print("Now working with robot: %s" % thymio_name)
    else:
        print(usage())
        sys.exit(1)
    thymio = BasicThymio(thymio_name)

    # Teleport the robot to a certain pose. If pose is different to the
    # origin of the world, you must account for a transformation between
    # odom and gazebo world frames.
    # NOTE: The goal of this step is *only* to show the available
    # tools. The launch file process should take care of initializing
    # the simulation and spawning the respective models

    #thymio.thymio_state_service_request([0.,0.,0.], [0.,0.,0.])
    #rospy.sleep(1.)

    thymio.move()
