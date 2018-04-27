#!/usr/bin/env python
import rospy
import sys

import numpy as np
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import Range
from nav_msgs.msg import Odometry
from math import cos, sin, asin, tan, atan2
# msgs and srv for working with the set_model_service
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty

# a handy tool to convert orientations
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class BasicThymio:

    def __init__(self, thymio_name):
        """init"""
        self.thymio_name = thymio_name
        rospy.init_node('basic_thymio_controller', anonymous=True)

        # Publish to the topic '/thymioX/cmd_vel'.
        self.velocity_publisher = rospy.Publisher(self.thymio_name + '/cmd_vel',
                                                  Twist, queue_size=10)

        self.center_buffer = [0.12]*8
	self.center_left_buffer = [0.12]*8
	self.center_right_buffer = [0.12]*8
	self.left_buffer = [0.12]*8
	self.right_buffer = [0.12]*8
	self.rear_right_buffer = [0.12]*8
	self.rear_left_buffer = [0.12]*8


        # A subscriber to the topic '/turtle1/pose'. self.update_pose is called
        # when a message of type Pose is received.
        self.pose_subscriber = rospy.Subscriber(self.thymio_name + '/odom',
                                                Odometry, self.update_state)
        self.sensor_center_subscriber = rospy.Subscriber(self.thymio_name+'/proximity/center', Range, self.callback_sensor_center)
        self.sensor_center_left_subscriber = rospy.Subscriber(self.thymio_name+'/proximity/center_left', Range, self.callback_sensor_center_left)
        self.sensor_center_right_subscriber = rospy.Subscriber(self.thymio_name+'/proximity/center_right', Range, self.callback_sensor_center_right)
        self.sensor_left_subscriber = rospy.Subscriber(self.thymio_name+'/proximity/left', Range, self.callback_sensor_left)
        self.sensor_right_subscriber = rospy.Subscriber(self.thymio_name+'/proximity/right', Range, self.callback_sensor_right)
        self.sensor_rear_right_subscriber = rospy.Subscriber(self.thymio_name+'/proximity/rear_right', Range, self.callback_sensor_rear_right)
        self.sensor_rear_left_subscriber = rospy.Subscriber(self.thymio_name+'/proximity/rear_left', Range, self.callback_sensor_rear_left)

        self.center_range = Range()
        self.center_left_range = Range()
        self.center_right_range = Range()
        self.left_range = Range()
        self.right_range = Range()
        self.rear_right_range = Range()
        self.rear_left_range = Range()


        self.current_pose = Pose()
        self.current_twist = Twist()
        # publish at this rate
        self.rate = rospy.Rate(10)

    def callback_sensor_center(self, data):
        self.center_range = data
        self.center_range.range = abs(round(self.center_range.range, 4))
        self.center_buffer.append(self.center_range.range)
        self.center_buffer.pop(0)
        #rospy.loginfo("State from Sensor Center: (%.4f) " % (self.center_range.range))

    def callback_sensor_center_left(self, data):
        self.center_left_range = data
        self.center_left_range.range = abs(round(self.center_left_range.range, 4))
        self.center_left_buffer.append(self.center_left_range.range)
        self.center_left_buffer.pop(0)
        #rospy.loginfo("State from Sensor Center Left: (%.4f) " % (self.center_left_range.range))

    def callback_sensor_center_right(self, data):
        self.center_right_range = data
        self.center_right_range.range = abs(round(self.center_right_range.range, 4))
        self.center_right_buffer.append(self.center_right_range.range)
        self.center_right_buffer.pop(0)
        #rospy.loginfo("State from Sensor Center Right: (%.4f) " % (self.center_right_range.range))

    def callback_sensor_left(self, data):
        self.left_range = data
        self.left_range.range = abs(round(self.left_range.range, 4))
        self.left_buffer.append(self.left_range.range)
        self.left_buffer.pop(0)
        #rospy.loginfo("State from Sensor Left: (%.4f) " % (self.left_range.range))

    def callback_sensor_right(self, data):
        self.right_range = data
        self.right_range.range = abs(round(self.right_range.range, 4))
        self.right_buffer.append(self.right_range.range)
        self.right_buffer.pop(0)
        #rospy.loginfo("State from Sensor Right: (%.4f) " % (self.right_range.range))

    def callback_sensor_rear_right(self, data):
        self.rear_right_range = data
        self.rear_right_range.range = abs(round(self.rear_right_range.range, 4))
        self.rear_right_buffer.append(self.rear_right_range.range)
        self.rear_right_buffer.pop(0)
        #rospy.loginfo("State from Sensor Rear Right: (%.4f) " % (self.rear_right_range.range))

    def callback_sensor_rear_left(self, data):
        self.rear_left_range = data
        self.rear_left_range.range = abs(round(self.rear_left_range.range, 4))
        self.rear_left_buffer.append(self.rear_left_range.range)
        self.rear_left_buffer.pop(0)
        #rospy.loginfo("State from Sensor Rear Left: (%.4f) " % (self.rear_left_range.range))

    def thymio_state_service_request(self, position, orientation):
        """Request the service (set thymio state values) exposed by
        the simulated thymio. A teleportation tool, by default in gazebo world frame.
        Be aware, this does not mean a reset (e.g. odometry values)."""
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            model_state = ModelState()
            model_state.model_name = self.thymio_name
            model_state.reference_frame = '' # the frame for the pose information
            model_state.pose.position.x = position[0]
            model_state.pose.position.y = position[1]
            model_state.pose.position.z = position[2]
            qto = quaternion_from_euler(orientation[0], orientation[0], orientation[0], axes='sxyz')
            model_state.pose.orientation.x = qto[0]
            model_state.pose.orientation.y = qto[1]
            model_state.pose.orientation.z = qto[2]
            model_state.pose.orientation.w = qto[3]
            # a Twist can also be set but not recomended to do it in a service
            gms = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            response = gms(model_state)
            return response
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def update_state(self, data):
        """A new Odometry message has arrived. See Odometry msg definition."""
        # Note: Odmetry message also provides covariance
        self.current_pose = data.pose.pose
        self.current_twist = data.twist.twist
        quat = (
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w)
        (roll, pitch, yaw) = euler_from_quaternion (quat)
        #rospy.loginfo("State from Odom: (%.5f, %.5f, %.5f) " % (self.current_pose.position.x, self.current_pose.position.y, yaw))

    def basic_move(self,state=[1]):
        """Moves the migthy thymio"""
        # Three states
        #   [1] moving forward in x, until is within 0.01 of wall
        #   [2] right after hitting wall, move back a little
        #   [3] rotate by <angle>
        vel_msg = Twist()
        vel_msg.linear.x = 0.08 # m/s
        vel_msg.angular.z = 0. # rad/s
        cur_time = start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown() : #and cur_time < start_time+10:
            if state[0] == 1:
                if np.mean(self.center_left_buffer+self.center_right_buffer) < 0.09:
                    state[0] = 2
                    vel_msg.linear.x = 0.0
                    self.velocity_publisher.publish(vel_msg)
                    #continue
                else:
                    # Publishing thymo vel_msg
                    self.velocity_publisher.publish(vel_msg)
            elif state[0] == 2:
                vel_msg.linear.x = 0.0
                self.velocity_publisher.publish(vel_msg)
                ang_speed = 0.1
                if np.mean(self.center_left_buffer) > np.mean(self.center_right_buffer):
                    ang_speed *= -1
                vel_msg.angular.z = ang_speed
                start_time = cur_time = rospy.Time.now().to_sec()
                while(not (np.allclose(np.mean(self.center_left_buffer),np.mean(self.center_right_buffer),atol=0.001) or np.allclose(np.mean(self.left_buffer),np.mean(self.right_buffer),atol=0.001))):
                    self.velocity_publisher.publish(vel_msg)
                    rospy.loginfo("FIRST loop, left: %.4f right: %.4f"%(np.mean(self.center_left_buffer),np.mean(self.center_right_buffer)))
                target_angle = np.pi/2.0
                start_time = cur_time = rospy.Time.now().to_sec()
                relative_angle = 0
                while(relative_angle < target_angle):
                    self.velocity_publisher.publish(vel_msg)
                    rospy.loginfo("SECOND loop")
                    cur_time = rospy.Time.now().to_sec()
                    relative_angle = np.abs(ang_speed)* (cur_time - start_time)
                vel_msg.angular.z = 0
                self.velocity_publisher.publish(vel_msg)




                #vel_msg.angular.z = angle_hat/2.0 # makes the turn in 2 secs
                #start_time = cur_time = rospy.Time.now().to_sec()
                #while(cur_time < start_time + 2):
                #    self.velocity_publisher.publish(vel_msg)
                #    self.rate.sleep()
                #    cur_time = rospy.Time.now().to_sec()
                vel_msg.angular.z = 0.0
                vel_msg.linear.x = 0.0
                break
            self.rate.sleep()
            cur_time = rospy.Time.now().to_sec()

        # Stop thymio. With is_shutdown condition we do not reach this point.
        vel_msg.linear.x = 0.
        vel_msg.angular.z = 0.
        self.velocity_publisher.publish(vel_msg)

        # waiting until shutdown flag (e.g. ctrl+c)
        #rospy.spin()

    def make_circle(self,right=1):
        est_time = 10*np.pi
        vel_msg = Twist()
        vel_msg.linear.x = 0.0 # m/s
        vel_msg.angular.z = 0. # rad/s
        angvel = 0.2 * right
        cur_time = start_time = rospy.Time.now().to_sec()
        while (not rospy.is_shutdown()) and (cur_time < (start_time + est_time)):
            # Publishing thymo vel_msg
            vel_msg.linear.x = 0.1
            cur_time = rospy.Time.now().to_sec()
            vel_msg.angular.z = angvel
            self.velocity_publisher.publish(vel_msg)
            # .. at the desired rate.
            self.rate.sleep()
        vel_msg.angular.z = 0.0
        vel_msg.linear.x = 0.0
        self.velocity_publisher.publish(vel_msg)


    def eight_move(self):
        est_time = 20*np.pi
        vel_msg = Twist()
        vel_msg.linear.x = 0.0 # m/s
        vel_msg.angular.z = 0. # rad/s
        cur_time = rospy.Time.now().to_sec() % est_time
        while not rospy.is_shutdown():
            # Publishing thymo vel_msg
            vel_msg.linear.x = 0.1
            cur_time = rospy.Time.now().to_sec() % est_time
            vel_msg.angular.z = 0.1
            self.velocity_publisher.publish(vel_msg)
            # .. at the desired rate.
            self.rate.sleep()

    def get_angle_from_proximity(self, center, center_left, left, center_right, right):
        theta = np.deg2rad(30) # angle between center and center_left/right
        phi = np.deg2rad(55) # angle between center and left/right
        angle = 90 # the angle we are trying to find, between x-axis of thymio (center prox sensor) and wall
        if center > 0.119:
            return None
        if center_left > 0.119 and center_right > 0.119:
            return None
        if center_left < center_right: # positive angle, wall closer to the left
            c = np.sqrt(center + center_left - 2*center*center_left*np.cos(theta)) # the distance between where the center prox sensor hits the wall and the center left prox sensor hits the wall
            angle = np.arcsin(center_left*np.sin(theta)/c)
            if left < 0.119 and False: # make the angle average calculated from 2 sensors for more accuracy
                c = np.sqrt(center + left - 2*center*left*np.cos(phi))
                angle += np.arcsin(left*np.sin(phi)/c)
                angle /= 2 # the average
            return angle
        else:
            c = np.sqrt(center + center_right - 2*center*center_right*np.cos(theta))
            angle = np.arcsin(center_right*np.sin(theta)/c)
            if ( right < 0.119) and False:
                c = np.sqrt(center + right - 2*center*right*np.cos(phi))
                angle += np.arcsin(right*np.sin(phi)/c)
                angle /= 2
            return -1*angle
        return angle


def usage():
    return "Wrong number of parameters. basic_move.py [thymio_name]"

if __name__ == '__main__':
    if len(sys.argv) == 2:
        thymio_name = sys.argv[1]
        print "Now working with robot: %s" % thymio_name
    else:
        print usage()
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

    thymio.basic_move()
    #thymio.make_circle(right=1)
    #thymio.make_circle(right=-1)
