import time
import gi
import os

import rospy
from std_msgs.msg import Float32MultiArray, UInt32MultiArray, UInt16MultiArray, UInt8MultiArray, UInt16, Int16MultiArray, String
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState, CompressedImage

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import math
import miro2 as miro

from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import radians 

import random


try:  # For convenience, import this util separately
    from miro2.lib import wheel_speed2cmd_vel  # Python 3
except ImportError:
    from miro2.utils import wheel_speed2cmd_vel  # Python 2
##########################


droop, wag, left_eye, right_eye, left_ear, right_ear = range(6)

def wiggle(v, n, m):
    v = v + float(n) / float(m)
    if v > 2.0:
        v -= 2.0
    elif v > 1.0:
        v = 2.0 - v
    return v


class meetingMiro:
    TICK = 0.02
    SLOW = 0.1  # Radial speed when turning on the spot (rad/s)
    FAST = 0.4  # Linear speed when kicking the ball (m/s)

    HEAD = 0 # Eye using to detect miro
    def __init__(self):

        self.cos_joints = Float32MultiArray()
        self.cos_joints.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.kin_joints = JointState()  # Prepare the empty message
        self.kin_joints.name = ["tilt", "lift", "yaw", "pitch"]

        self.input_package = None
        
        # robot name
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

        # publishers
        self.pub_cmd_vel = rospy.Publisher(topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0)
        self.pub_cos = rospy.Publisher(topic_base_name + "/control/cosmetic_joints", Float32MultiArray, queue_size=0)
        self.pub_kin = rospy.Publisher(topic_base_name + "/control/kinematic_joints", JointState, queue_size=0)
        self.pub_animal_state = rospy.Publisher(topic_base_name+"/core/animal/state", miro.msg.animal_state,queue_size=0)
    
    def callback_package(self, msg):
        # store for processing in update_gui
        self.input_package = msg
    

    def blink(self,*args):
            self.cos_joints.data[left_eye] = wiggle(0.0, 1, 1)
            self.cos_joints.data[right_eye] = wiggle(0.0, 1, 1)
            self.pub_cos.publish(self.cos_joints)
            rospy.sleep(0.2)
            self.cos_joints.data[left_eye] = wiggle(1, 1, 1)
            self.cos_joints.data[right_eye] = wiggle(1, 1, 1)
            self.pub_cos.publish(self.cos_joints)
            rospy.sleep(0.2)
    
    def earMove(self,*args):
        self.cos_joints.data[left_ear] = wiggle(0.3333,1,1)
        self.cos_joints.data[right_ear] = wiggle(0.3333,1,1)
        self.pub_cos.publish(self.cos_joints)
        rospy.sleep(0.2)

    def earMoveBack(self,*args):
        self.cos_joints.data[left_ear] = wiggle(1.9,1,1)
        self.cos_joints.data[right_ear] = wiggle(1.9,1,1)
        self.pub_cos.publish(self.cos_joints)
        rospy.sleep(0.2)

    def neckMovement(self,*args):
        self.kin_joints.position = [0.0, radians(50), yawV, 0.0]

    def doubleBlink(self, *args):
        self.blink()
        self.blink()

    
    def drive(self, speed_l=0.1, speed_r=0.1):  # (m/sec, m/sec)
        """
        Wrapper to simplify driving MiRo by converting wheel speeds to cmd_vel
        """
        # Prepare an empty velocity command message
        msg_cmd_vel = TwistStamped()

        # Desired wheel speed (m/sec)
        wheel_speed = [speed_l, speed_r]

        # Convert wheel speed to command velocity (m/sec, Rad/sec)
        (dr, dtheta) = wheel_speed2cmd_vel(wheel_speed)

        # Update the message with the desired speed
        msg_cmd_vel.twist.linear.x = dr
        msg_cmd_vel.twist.angular.z = dtheta

        # Publish message to control/cmd_vel topic
        self.pub_cmd_vel.publish(msg_cmd_vel)

    def turn_slow(self):
        count = 0
        while count<7000:
            count += 1
            self.drive(-self.SLOW, self.SLOW)


    #def head_down(self):

    def reset_head_pose(self):
        """
        Reset MiRo head to default position, to avoid having to deal with tilted frames
        """
        self.kin_joints = JointState()  # Prepare the empty message
        self.kin_joints.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin_joints.position = [0.0, radians(45.0), 0.0, 0.0]
        t = 0
        HEAD = 0
        while not rospy.core.is_shutdown():  # Check ROS is running
            # Publish state to neck servos for 1 sec
            self.pub_kin.publish(self.kin_joints)
            rospy.sleep(self.TICK)
            t += self.TICK
            if t > 1:
                break

    def happy_voice(self):
        count = 0
        msg = miro.msg.animal_state()
        # loop
        while count<100:
            count += 1

            # set emotion
            msg.emotion.valence = 1.0 # 0.0 = sad, 1.0 = happy
            msg.emotion.arousal = 1.0 # 0.0 = low, 1.0 = high

            # set high ambient sound level to maximize volume
            # (see animal_state.msg for more details)
            msg.sound_level = 0.1

            # wakefulness also used as audio gain
            msg.sleep.wakefulness = 1.0

            # enable voice
            msg.flags = miro.constants.ANIMAL_EXPRESS_THROUGH_VOICE

            # update voice node
            self.pub_animal_state.publish(msg)

            # state
            time.sleep(0.3)

    def wagging(self):
        count = 0
        while count < 40:
            count += 1
            if count%2 == 0:
                self.cos_joints.data[wag] = 1.0
            else:
                self.cos_joints.data[wag] = 0.0
            self.pub_cos.publish(self.cos_joints)
            rospy.sleep(0.1)

    def miro_deteted(self):
        msg = miro.msg.animal_state()
        count = 0
        while count < 40:
            count += 1
            # set emotion
            msg.emotion.valence = 1.0 # 0.0 = sad, 1.0 = happy
            msg.emotion.arousal = 1.0 # 0.0 = low, 1.0 = high

            # set high ambient sound level to maximize volume
            # (see animal_state.msg for more details)
            msg.sound_level = 0.1

            # wakefulness also used as audio gain
            msg.sleep.wakefulness = 1.0

            # enable voice
            msg.flags = miro.constants.ANIMAL_EXPRESS_THROUGH_VOICE

            # update voice node
            self.pub_animal_state.publish(msg)

            if count%2 == 0:
                self.cos_joints.data[wag] = 1.0
            else:
                self.cos_joints.data[wag] = 0.0
            self.pub_cos.publish(self.cos_joints)


            time.sleep(0.3)

    def driveForward(self):
        count = 0
        while count < 40000:
            count += 1
            self.drive(self.FAST, self.FAST)

    def headMovement(self):
        if self.HEAD == 2:
            self.kin_joints.position = [0.0, radians(50.0), -0.5, 0.0]
            self.HEAD = 1
        else:
            self.kin_joints.position = [0.0, radians(50.0), 0.5, 0.0]
            self.HEAD = 2
        self.pub_kin.publish(self.kin_joints)