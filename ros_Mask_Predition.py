import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from cv_bridge import CvBridge, CvBridgeError  

from PIL import Image as Img
import cv2
import albumentations as A

import miro2 as miro

import time
import os
from tqdm.notebook import tqdm

from torchsummary import summary
import segmentation_models_pytorch as smp

import rospy
import rosnode
from sensor_msgs.msg import JointState, BatteryState, Image, Imu, Range, CompressedImage
from std_msgs.msg import Float32MultiArray, UInt32MultiArray, UInt16MultiArray, UInt8MultiArray, UInt16, Int16MultiArray, String

import threading

import meetingMiro as mri

MODEL_PATH = "/home/dengyun/Semantic_SegmentationModel_5.pt"

MTX = np.array([[1.47195671e+03, 0.00000000e+00, 7.27986326e+02],
 [0.00000000e+00, 1.24393214e+03, 3.47346236e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
 ])
DIST = np.array([[-0.70642406,-2.63680189 ,0.00831157,-0.07746413 ,6.71721589]])

class MiroDetector:
    NODE_EXISTS = False 

    def __init__(self):
        print('loading model...')
        self.model = torch.load(MODEL_PATH,map_location=torch.device('cpu'))
        self.image_preprocess = A.Resize(768, 1152, interpolation=cv2.INTER_NEAREST)

        self.illum = UInt32MultiArray()
        self.illum.data = [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF]

        print('initializing ros core...')
        name = 'ros_Mask_Prediction'
        # Initialise ROS node ('disable_rostime=True' needed to work in PyCharm)
        self.image_converter = CvBridge()
        if not self.NODE_EXISTS:
            rospy.init_node(name, anonymous=True)

        # ROS topic root
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")


        self.input_package = None
        self.input_camera = [None, None]

        self.pub_illum = rospy.Publisher(topic_base_name + "/control/illum", UInt32MultiArray, queue_size=0)
        self.pub_command = rospy.Publisher(topic_base_name + "/control/command", String, queue_size=0)

        self.sub_caml = rospy.Subscriber(
            topic_base_name + "/sensors/caml/compressed",
            CompressedImage,
            self.callback_caml,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.sub_camr = rospy.Subscriber(
            topic_base_name + "/sensors/camr/compressed",
            CompressedImage,
            self.callback_camr,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.sub_package = rospy.Subscriber(topic_base_name + "/sensors/package",
            miro.msg.sensors_package, self.callback_package, queue_size=1, tcp_nodelay=True)

        
        self.miro_behavior = mri.meetingMiro()
        self.miro_behavior.reset_head_pose()
        self.status_code = 0
        self.miro_distance = 0
        print('initialized successfully')

    @staticmethod
    def ros_sleep(time):
        # Sleep after init to prevent accessing data before a topic is subscribed
        rospy.sleep(time)

    def predict_mask(self, image):
        image = self.image_preprocess(image = image)['image']
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        self.model.eval()
        t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        image = t(image)
        self.model.to('cpu') 
        image=image.to('cpu')
        
        with torch.no_grad():
            image = image.unsqueeze(0)
            output = self.model(image)
            masked = torch.argmax(output, dim=1)
            masked = masked.cpu().squeeze(0)

        return masked

    def callback_caml(self, ros_image):  # Left camera
        self.callback_cam(ros_image, 0)

    def callback_camr(self, ros_image):  # Right camera
        self.callback_cam(ros_image, 1)

    def callback_cam(self, ros_image, index):
        """
        Callback function executed upon image arrival
        """
        # Silently(-ish) handle corrupted JPEG frames
        try:
            # Convert compressed ROS image to raw CV image
            image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "rgb8")
            # Store image as class attribute for further use
            self.input_camera[index] = image
        except CvBridgeError as e:
            # Ignore corrupted frames
            pass

    def callback_package(self, msg):
        self.input_package = msg
    
    def LED_ColorChange(self,red = 0.0,green = 0.0,blue = 0.0):
        color_detail = (int(red),int(green),int(blue))
        color = '0xFF%02x%02x%02x'%color_detail
        color = int(color,16)
        self.illum.data = [color,color,color,color,color,color]

    def pub_color(self):
        self.pub_illum.publish(self.illum)

    def LED_Flashing(self):
        st = time.time()
        while time.time()-st < 2:
            self.LED_ColorChange(red = 100)
            self.pub_color()
            self.LED_ColorChange()
            self.pub_color()


    def detect_miro(self,locking_miro = False):
        start_time = time.time()
        counter = 0
        while 1:
            rospy.sleep(0.5)#wait till the movement ends and the image does not blur
            
            image = self.input_camera[0]

            mod_l = cv2.undistort(image, MTX, DIST, None)
            mask = self.predict_mask(mod_l)
            roated_image = np.rot90(mask,1)
            
            #Image display
            '''
            fig, (ax1, ax2,ax3,ax4) = plt.subplots(1,4, figsize=(20,10))
            ax1.imshow(image)
            ax1.set_title('image')

            ax2.imshow(mod_l)
            ax2.set_title('undistort image')


            ax3.imshow(mask)
            ax3.set_title('Predict Mask')

            print(roated_image.shape)
            white_pixels = np.array(np.where(roated_image == 1))
            first_white_pixel = white_pixels[:,0]
            last_white_pixel = white_pixels[:,-1]
            
            print(first_white_pixel)
            print(last_white_pixel)

            ax4.imshow(roated_image)
            ax4.set_title('rotated image')
            ax4.set_axis_off()

            plt.show()
            '''

            
            if locking_miro:
                return mask
            
            if mask.max() == 1:#Miro found, go into next step
                print('Miro found!')
                self.LED_ColorChange(green = 100)
                self.pub_color()
                self.miro_behavior.reset_head_pose()
                self.miro_behavior.doubleBlink()
                self.status_code = 2
                self.miro_distance = mask.sum()
                print(self.miro_distance)
                break
            elif counter%8 != 0:#Miro not found, turn around and keep searching
                print('Miro not found!')
                print('continue searching')
                self.miro_behavior.turn_slow()
            else:
                self.miro_behavior.driveForward()
            
            counter += 1
                

        return mask

    def lock_onto_miro(self,image):
        '''
        keep Miro face toward the other miro
        left camera center is 285
        '''

        roated_image = np.rot90(image,1)

        white_pixels = np.array(np.where(roated_image == 1))
        rightmost = white_pixels[:,0][0]
        leftmost = white_pixels[:,-1][0]

        print(leftmost)
        print(rightmost)

        if rightmost<=285 and leftmost<=576:# Miro is facing toward the other miro
            self.status_code = 3
        elif leftmost>285 and leftmost<=576: # Miro is on the right side 
            self.miro_behavior.drive(0.3,-0.3)
            self.status_code = 3
        else:
            self.miro_behavior.drive(-0.3,0.3)
            self.status_code = 2


    def loop(self):
        touch_count = 0
        wait_count =0
        hostwait = 1
        cmd = "frame=720w@15"
        self.pub_command.publish(cmd)
        self.miro_behavior.reset_head_pose()
        self.miro_behavior.driveForward()
        

        while hostwait:
            if self.input_package != None:
                hostwait = 0
        while hostwait == 0:
            # Step 1. Find Miro
            if self.status_code == 1:
                self.detect_miro()
            # Step 2. Orient towards it
            elif self.status_code == 2:
                self.lock_onto_miro(self.detect_miro(locking_miro = True))

            # Step 3: Drive Forward
            elif self.status_code == 3:
                self.miro_behavior.driveForward()
                self.miro_behavior.miro_deteted()

            else:
                self.status_code = 1

            # Yield
            rospy.sleep(0.02)

                

    
if __name__ == "__main__":
    main = MiroDetector()
    main.loop()

