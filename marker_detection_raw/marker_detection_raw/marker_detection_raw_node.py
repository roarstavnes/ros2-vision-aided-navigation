import rclpy
import cv2
import numpy as np
import math

from rclpy.node import Node
from marker_detection_raw.kinematics import *

# Ros messages
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose

# Bridge between ROS and OpenCV
from cv_bridge import CvBridge

# Instantiate CvBridge
bridge = CvBridge()

pi = math.pi

class Aruco:
    def __init__(self, id, marker_length, x, y, z, phi, theta, psi):
        self.id             = id
        self.length         = marker_length
        self.position       = np.array([[x, y, z]]).T
        self.orientation    = np.array([[phi, theta, psi]]).T
        self.quat           = euler2q(self.orientation)
        self.R_ned2marker   = Rzyx(self.orientation)
        self.H_ned2marker   = H(self.R_ned2marker, self.position)
        self.H_marker2ned   = Hinv(self.R_ned2marker, self.position)

# Create a list for all markers
markers = []

# Marker location for the yellow subsea-unit
markers.append( Aruco(110, 0.15, 0.19, 0.0, 0.202, 0, 0, 0)) 
markers.append( Aruco(29, 0.15, 0.932, 0.0, 0.401, 0, 0, 0))
markers.append( Aruco(82, 0.15, 0.545, 0.0, 0.702, 0, 0, 0))


class markerDetectionRaw(Node):
    def __init__(self):
        super().__init__('marker_detection')
        self.image = None
        self.camera_matrix = np.array([ [454.30525008,  0.,         347.30162363],
                                        [0.,        453.72480994,   228.78599989],
                                        [0.,        0.,         1]])
        self.distortion_coefficients = np.array([[-0.00448644,  0.01318547,  0.00028959,  0.00493763, 0]])
        self.dt = 0.1
   

        self.img_pose_pub = self.create_publisher(
            PoseArray,
            '/camera/poses',
            1
        )

        self.detect_markers_pub = self.create_publisher(
            Image,
            '/camera/detected_markers',
            1
        )


        gst_str = 'udpsrc port=2000 ! application/x-rtp, encoding-name=(string)H264, payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink'
        
        self.cap = cv2.VideoCapture(gst_str)

        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

        self.timer = self.create_timer(0.5, self.run)

    
    def run(self):
        ret, self.image = self.cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ....")
            return
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        aruco_dict      = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
        aruco_params    = cv2.aruco.DetectorParameters_create()

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        # cv2.aruco.drawDetectedMarkers(self.image, corners, ids=None)
        if corners != []:
            msg = PoseArray()
            for i in range(0,len(ids)):
                for marker in markers:
                    if ids[i] == marker.id:
                        rvecs, tvecs, temp = cv2.aruco.estimatePoseSingleMarkers(corners[i],marker.length,cameraMatrix=self.camera_matrix,distCoeffs=self.distortion_coefficients,rvecs=None,tvecs=None)

                        t_x = tvecs[0][0][0]
                        t_y = tvecs[0][0][1]
                        t_z = tvecs[0][0][2]
                        o_camera2marker = np.array([[t_x, t_y, t_z]]).T

                        r_x = rvecs[0][0][0] 
                        r_y = rvecs[0][0][1] 
                        r_z = rvecs[0][0][2]
                        r   = np.array([[r_x, r_y, r_z]]).T  
                        R_camera2marker = Rzyx(r)
                        quat  = R2quat(R_camera2marker)

                        # Publish marker pose estimate
                        msg_camera = Pose()
                        msg_camera.position.x    = t_x
                        msg_camera.position.y    = t_y
                        msg_camera.position.z    = t_z
                        msg_camera.orientation.w = quat[0][0]
                        msg_camera.orientation.x = quat[1][0]
                        msg_camera.orientation.y = quat[2][0]
                        msg_camera.orientation.z = quat[3][0]

                        msg_marker = Pose()
                        msg_marker.position.x     = float(marker.position[0][0])
                        msg_marker.position.y     = float(marker.position[1][0])
                        msg_marker.position.z     = float(marker.position[2][0])
                        msg_marker.orientation.w  = marker.quat[0][0]
                        msg_marker.orientation.x  = marker.quat[1][0]
                        msg_marker.orientation.y  = marker.quat[2][0]
                        msg_marker.orientation.z  = marker.quat[3][0]

                        msg.poses.append(msg_camera)
                        msg.poses.append(msg_marker)

            #Publish messasge           
            msg.header.stamp = self.get_clock().now().to_msg()
            self.img_pose_pub.publish(msg)           
                                
        
        
        # imgmsg = Image()
        # imgmsg = bridge.cv2_to_imgmsg(self.image,"bgr8")
        # self.detect_markers_pub.publish(imgmsg)
        #self.image = None

    # Image callback if camera image is sent as a ros message
    # def img_callback(self,msg):
    #     self.image = bridge.imgmsg_to_cv2(msg, "bgr8")
    #     #self.run()

def main():
    rclpy.init()

    marker_node = markerDetectionRaw()

    rclpy.spin(marker_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    marker_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

