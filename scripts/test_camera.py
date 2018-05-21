#!/usr/bin/env python

from __future__ import print_function

import roslib; roslib.load_manifest('teleop_twist_keyboard')
import rospy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import numpy as np
import cv2, cv_bridge

import sys, select, termios, tty, os

msg = """
Reading from the keyboard  and Publishing to Twist!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

For Holonomic mode (strafing), hold down the shift key:
---------------------------
   U    I    O
   J    K    L
   M    <    >

t : up (+z)
b : down (-z)

anything else : stop

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%

CTRL-C to quit
"""

moveBindings = {
		'w':(0.2,0),
		's':(0,0),
		'a':(0.3,0.6),
		'd':(0.3,-0.6),
		'q':(0.3,1.0),
		'e':(0.3,-1.0),
		'c':(-0.15,0),
                'r':(0,0.5),
                't':(0,-0.5)
    	       }

oveBindings = {
		'i':(0.2,0),
		'o':(0,0),
		'j':(0.3,0.6),
		'l':(0.3,-0.6),
		'u':(0.3,0.8),
		'o':(0.3,-0.8),
		'm':(-0.2,0),
                'r':(0,0.5),
                't':(0,-0.5)
    	       }

speedBindings={
		'q':(1.1,1.1),
		'z':(.9,.9),
		'w':(1.1,1),
		'x':(.9,1),
		'e':(1,1.1),
		'c':(1,.9),
	      }

recording = False
recording_counter = 0
train_dir = 'dummy_dir'

global_lin = 0
global_ang = 0

bridge = cv_bridge.CvBridge()



def getKey():
	tty.setraw(sys.stdin.fileno())
	select.select([sys.stdin], [], [], 0)
	key = sys.stdin.read(1)
	termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
	return key


def vels(speed,turn):
	return "currently:\tspeed %s\tturn %s " % (speed,turn)

def callback_image(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except cv_bridge.cvbridgeerror as e:
        print(e)

    (rows,cols,channels) = cv_image.shape
    if cols > 60 and rows > 60 :
        cv2.circle(cv_image, (50,50), 10, 255)

    cv2.imshow("frame", cv_image)
    # press q on keyboard to  exit
    if cv2.waitKey(15) & 0xff == ord('q'):
        exit(1)

def callback_image_save(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except cv_bridge.cvbridgeerror as e:
        print(e)

    (rows,cols,channels) = cv_image.shape
    if cols > 60 and rows > 60 :
        cv2.circle(cv_image, (50,50), 10, 255)
    global recording
    global recording_counter
    global train_dir
    recording_counter += 1
    if recording and recording_counter % 2 == 0:
        filename = os.path.join(train_dir,'{0}_lin{1}_ang{2}.png'.format(recording_counter,global_lin,global_ang))
        cv2.imwrite(filename,cv_image)
        



if __name__=="__main__":
    	settings = termios.tcgetattr(sys.stdin)
        if len(sys.argv) == 2:
            thymio_name = sys.argv[1]
        else:
            print('Specify thymio name, defaulting to thymio10')
            thymio_name = 'thymio10'

	pub = rospy.Publisher('/{0}/cmd_vel'.format(thymio_name), Twist, queue_size = 1)
        sensor_center_subscriber = rospy.Subscriber('/{0}/camera/image_raw'.format(thymio_name), Image, callback_image_save)
	rospy.init_node('teleop_twist_keyboard')

	speed = rospy.get_param("~speed", 0.2)
	turn = rospy.get_param("~turn", 1.0)
	x = 0
	y = 0
	z = 0
	th = 0
	status = 0

	try:
		print(msg)
		twist = Twist()
                lin_speed = 0
		print(vels(speed,turn))
		while(1):
			key = getKey()
                        try:
                            lin, ang = moveBindings[key]
                        except:
                            if key == 'p':
                                import time
                                train_dir = time.strftime("%Y%m%d-%H%M%S")
                                os.makedirs(train_dir)
                                recording = True
                                recording_counter = 0
                            elif key == 'o':
                                recording = False
		            lin = 0
                            ang = 0
                            if (key == '\x03'):
			        break
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

    		termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


