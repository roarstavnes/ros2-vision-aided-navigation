import rclpy
import threading
import os
import math

import numpy as np
from std_msgs.msg import String
from rclpy.node import Node
from kf_attitude.kinematics import *
from rcl_interfaces.msg import ParameterType
from sensor_msgs.msg import Imu, MagneticField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray
from concurrent.futures import ThreadPoolExecutor
from rclpy.executors import Executor
from rclpy.exceptions import ParameterNotDeclaredException


class kf(Node):
    def __init__(self):
        super().__init__('kf')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('Qd', None),
                ('Rd', None),
                ('Tacc', None),
                ('Tars', None),
                ('dt', None)
            ]
        )


        # Measurements
        self.f_imu  = np.array([[1.0, 1.0, 1.0]]).T
        self.w_imu  = np.array([[1.0, 1.0, 1.0]]).T
        self.m_imu  = np.array([[1.0, 1.0, 1.0]]).T
        self.poses  = None
        self.receivedCameraMeasurement = False

        # Qualisys
        self.quat_qualisys          = np.zeros((4,1))
        self.quat_qualisys[0][0]    = 1
        self.m_qualisys             = np.array([[1.0, 1.0, 1.0]]).T
        self.m_qualisys_ref         = np.array([[1.0, 0.0, 0.0]]).T
        self.counter = 0.0
        self.runtime = 0.0
        # Magnetometer reference vector
        self.m_ref  = np.array([[-21.9, 18.5, 14.5]]).T

        # Transformation matrix from body to camera
        pi = math.pi
        self.d_b2c  = np.array([[0.14, 0, 0]]).T
        euler       = np.array([[pi/2, 0.0, pi/2]]).T
        eulercorrection = np.array([[0.0, 2*pi/180, 1.5*pi/180]]).T
        self.R_b2c  = Rzyx(euler).dot(Rzyx(eulercorrection))

        # Transformation matrix from body to sensor frame
        self.d_b2s = np.array([[0.085, 0.04, 0.0]]).T
        self.R_b2s = np.array([ [ 0.9999,  0.0,    -0.0132],
                                [-0.0007,  0.9984, -0.0560],
                                [ 0.0132,  0.0560,  0.9983]])

        # Kalman filter parameters
        self.p_ins      = np.zeros((3,1))
        self.v_ins      = np.zeros((3,1))
        self.b_acc_ins  = np.zeros((3,1))
        self.quat_ins   = np.zeros((4,1))
        self.quat_ins[0][0] = 1
        self.b_ars_ins  = np.zeros((3,1))

        Qd = self.get_parameter('Qd').get_parameter_value().double_array_value
        self.Rd = self.get_parameter('Rd').get_parameter_value().double_array_value
        self.Q_d = np.diag(Qd)

        self.P_prd = np.identity(15)
        self.P_hat  = np.zeros((15,15))

        self.T_acc = self.get_parameter('Tacc').get_parameter_value().double_value
        self.T_ars = self.get_parameter('Tars').get_parameter_value().double_value

        self.g_n = np.array([[0, 0, 9.8163]]).T


        self.dt = self.get_parameter('dt').get_parameter_value().double_value 
        
        self.odom_sub = self.create_subscription(
            PoseArray,
            '/camera/poses', 
            self.pose_callback,
            1
        )
        self.odom_sub # prevent unused variable warning

        self.imu_sub = self.create_subscription(
            Imu,
            '/bno055/imu_raw',
            self.imu_callback,
            1
        )
        self.imu_sub

        self.mag_sub = self.create_subscription(
            MagneticField,
            '/bno055/mag',
            self.mag_callback,
            1
        )
        self.mag_sub

        self.qualisys_sub = self.create_subscription(
            Odometry,
            '/qualisys/LV/odom',
            self.qualisys_callback,
            1
        )

        self.x_ins_pub = self.create_publisher(
            Odometry,
            '/observer/odom',
            1
        )

        self.timer = self.create_timer(self.dt, self.run)


    def run(self):
        
        Z_3 = np.zeros((3,3))
        I_3 = np.identity(3)

        # Jacobian matrices
        R = Rquat(self.quat_ins)
        T = Tquat(self.quat_ins)

        # Bias compensated IMU measurements
        f_ins = self.R_b2s.T.dot(self.f_imu - self.b_acc_ins)
        w_ins = self.w_imu - self.b_ars_ins

        # Normalized gravity vectors
        v10 = np.array([[0, 0, 1]]).T
        v1  = - f_ins
        v1  = v1/np.linalg.norm(v1)

        # Normalized magnetic field vectors
        #v20 = self.m_ref/np.linalg.norm(self.m_ref)
        #v2  = self.m_imu/np.linalg.norm(self.m_ref)
        v20 = self.m_qualisys_ref
        v2  = self.m_qualisys/np.linalg.norm(self.m_qualisys)

        # Define state space matrices
        A = np.concatenate((np.concatenate((Z_3,    I_3,    Z_3,                    Z_3,                    Z_3),axis=1),
                            np.concatenate((Z_3,    Z_3,   -1*R,                   -1*R.dot(S(f_ins)),      Z_3),axis=1),
                            np.concatenate((Z_3,    Z_3,   -(1/self.T_acc)*I_3,     Z_3,                    Z_3),axis=1),
                            np.concatenate((Z_3,    Z_3,    Z_3,                   -1*S(w_ins),            -I_3),axis=1),
                            np.concatenate((Z_3,    Z_3,    Z_3,                    Z_3,                   -(1/self.T_ars)*I_3),axis=1)),axis=0)
                            

        Ad = np.identity(15) + self.dt*A

        Cd = np.concatenate((np.concatenate((I_3,   Z_3,    Z_3,    Z_3,                Z_3),axis=1),
                             np.concatenate((Z_3,   Z_3,    Z_3,    S(R.T.dot(v10)),    Z_3),axis=1),
                             np.concatenate((Z_3,   Z_3,    Z_3,    S(R.T.dot(v20)),    Z_3),axis=1)),axis=0)

        Ed = self.dt*np.concatenate((np.concatenate((Z_3,     Z_3,    Z_3,      Z_3),axis=1),   
                                     np.concatenate((-1*R,    Z_3,    Z_3,      Z_3),axis=1),
                                     np.concatenate((Z_3,     I_3,    Z_3,      Z_3),axis=1),
                                     np.concatenate((Z_3,     Z_3,    -1*I_3,   Z_3),axis=1),
                                     np.concatenate((Z_3,     Z_3,    Z_3,      I_3),axis=1)),axis=0)
  
        # Check if aiding measurement is available
        if self.receivedCameraMeasurement == False and self.counter <= 0.1:
            self.P_hat = self.P_prd
        else:
            self.counter = 0
            eps_g       = v1 - R.T.dot(v10)
            eps_mag     = v2 - R.T.dot(v20)

            if self.receivedCameraMeasurement == True and self.runtime >= 10:

                self.receivedCameraMeasurement == False
                Cd = np.concatenate((np.concatenate((I_3,   Z_3,    Z_3,    Z_3,                Z_3),axis=1),
                                     np.concatenate((Z_3,   Z_3,    Z_3,    S(R.T.dot(v10)),    Z_3),axis=1),
                                     np.concatenate((Z_3,   Z_3,    Z_3,    S(R.T.dot(v20)),    Z_3),axis=1)),axis=0)   
                R_d = np.diag(self.Rd)

                # Take the weighted discounted average for the translation between camera and the marker(s)
                weight  = 0
                y_pos   = np.zeros((3,1)) 
                
                for i in range(0,int(len(self.poses)/2)):
                    t_x = self.poses[2*i].position.x
                    t_y = self.poses[2*i].position.y
                    t_z = self.poses[2*i].position.z 
                    t   = np.array([[t_x, t_y, t_z]]).T

                    x_i = self.poses[2*i+1].position.x
                    y_i = self.poses[2*i+1].position.y
                    z_i = self.poses[2*i+1].position.z
                    p_i = np.array([[x_i, y_i, z_i]]).T

                    y_i = self.R_b2c.dot(t) + self.d_b2c - self.d_b2s

                    norm_inv = 1/np.sqrt(y_i[0][0]**2 + y_i[1][0]**2 + y_i[2][0]**2)

                    if norm_inv > 1:
                        norm_inv = 1

                    weight += norm_inv
                    
                    y_pos += norm_inv*(p_i - R.dot(y_i))

                y_pos   = y_pos/weight
                
                eps_pos = y_pos - self.p_ins
                eps     = np.concatenate((eps_pos, eps_g, eps_mag), axis=0)
            
            else:
                Cd = np.concatenate((np.concatenate((Z_3,   Z_3,    Z_3,    S(R.T.dot(v10)),    Z_3),axis=1),
                                     np.concatenate((Z_3,   Z_3,    Z_3,    S(R.T.dot(v20)),    Z_3),axis=1)),axis=0)   
                eps = np.concatenate((eps_g,eps_mag),axis = 0)
                R_d  = np.diag(self.Rd[3:9])
            # KF gain: K[k]
            K   = self.P_prd.dot(Cd.T).dot(np.linalg.inv(Cd.dot(self.P_prd).dot(Cd.T) + R_d))
            IKC = np.identity(15) - K.dot(Cd)

            # Corrector 
            delta_x_hat = K.dot(eps)
            self.P_hat  = IKC.dot(self.P_prd).dot(IKC.T) + K.dot(R_d).dot(K.T)

            # Error quaternion
            delta_a     = delta_x_hat[9:12]
            delta_quat_hat = 1/np.sqrt(4 + delta_a.T.dot(delta_a)) * np.array([[2, delta_a[0][0], delta_a[1][0], delta_a[2][0]]]).T

            # INS reset: x_ins[k]
            self.p_ins      += delta_x_hat[:3]
            self.v_ins      += delta_x_hat[3:6]
            if self.receivedCameraMeasurement is True: 
                self.b_acc_ins  += delta_x_hat[6:9]
                self.b_ars_ins  += delta_x_hat[12:15]
            self.quat_ins   = quatprod(self.quat_ins,delta_quat_hat)
            self.quat_ins   = self.quat_ins/np.linalg.norm(self.quat_ins)

        # Predictor: P_prd[k+1]
        self.P_prd = Ad.dot(self.P_hat).dot(Ad.T) + Ed.dot(self.Q_d).dot(Ed.T)

        # INS propagation: x_ins[k+1]
        self.p_ins      += self.dt*self.v_ins
        self.v_ins      += self.dt*(R.dot(f_ins) + self.g_n)
        self.quat_ins   += self.dt*T.dot(w_ins)
        self.quat_ins    = self.quat_ins/np.linalg.norm(self.quat_ins)
        self.counter    += self.dt
        self.runtime    += self.dt


        # Publish estimate
        d_pos = R.dot(-self.d_b2s)

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = self.p_ins[0][0] + d_pos[0][0]
        msg.pose.pose.position.y = self.p_ins[1][0] + d_pos[1][0]
        msg.pose.pose.position.z = self.p_ins[2][0] + d_pos[2][0]
        msg.pose.pose.orientation.w = self.quat_ins[0][0]
        msg.pose.pose.orientation.x = self.quat_ins[1][0]
        msg.pose.pose.orientation.y = self.quat_ins[2][0]
        msg.pose.pose.orientation.z = self.quat_ins[3][0]
        msg.twist.twist.linear.x = self.v_ins[0][0]
        msg.twist.twist.linear.y = self.v_ins[1][0]
        msg.twist.twist.linear.z = self.v_ins[2][0]
        self.x_ins_pub.publish(msg)

    def pose_callback(self,msg):
        self.receivedCameraMeasurement = True
        self.poses = msg.poses

    def imu_callback(self,msg):
        self.f_imu[0][0] = msg.linear_acceleration.x
        self.f_imu[1][0] = msg.linear_acceleration.y
        self.f_imu[2][0] = msg.linear_acceleration.z
        self.w_imu[0][0] = msg.angular_velocity.x
        self.w_imu[1][0] = msg.angular_velocity.y 
        self.w_imu[2][0] = msg.angular_velocity.z

    
    def mag_callback(self,msg):
        self.m_imu[0][0] = msg.magnetic_field.x
        self.m_imu[1][0] = msg.magnetic_field.y
        self.m_imu[2][0] = msg.magnetic_field.z

    def qualisys_callback(self,msg):
        self.quat_qualisys[0][0] = msg.pose.pose.orientation.w
        self.quat_qualisys[1][0] = msg.pose.pose.orientation.x
        self.quat_qualisys[2][0] = msg.pose.pose.orientation.y
        self.quat_qualisys[3][0] = msg.pose.pose.orientation.z

        R = Rquat(self.quat_qualisys).T
        self.m_qualisys[0][0] = R[0][0]
        self.m_qualisys[1][0] = R[1][0]
        self.m_qualisys[2][0] = R[2][0]


def main():
    rclpy.init()

    kf_node = kf()

    rclpy.spin(kf_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    kf_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
