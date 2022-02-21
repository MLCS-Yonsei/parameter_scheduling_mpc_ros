#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
import time
import numpy as np
from tf import TransformListener
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from casadi import *
from mdp import MarkovDecisionProcess
from nmpc_cartesian import NmpcCartesian
from nmpc_polar import NmpcPolar
from nmpc_switching import NmpcSwitching
from utils import normalize_angle
EPS = np.finfo(float).eps



class ParameterSchedulingMPC:

    def __init__(self):

        self.isInitialized = False
        rospy.init_node('parameter_scheduling_mpc', anonymous=False)
        time.sleep(1)
        self.tic = None
        self.toc = None

        self.loop_rate = rospy.get_param("/parameter_scheduling_mpc/loop_rate")
        self.use_odom = rospy.get_param("/parameter_scheduling_mpc/use_odom")
        self.regularization_u = rospy.get_param("/parameter_scheduling_mpc/regularization_u")
        self.regularization_v = rospy.get_param("/parameter_scheduling_mpc/regularization_v")
        self.regularization_omega = rospy.get_param("/parameter_scheduling_mpc/regularization_omega")
        self.T = rospy.get_param("/parameter_scheduling_mpc/step_time")
        self.N = rospy.get_param("/parameter_scheduling_mpc/time_horizon")
        self.w_max = rospy.get_param("/parameter_scheduling_mpc/max_wheel_angular_vel")
        self.a_max = rospy.get_param("/parameter_scheduling_mpc/max_wheel_angular_acc")
        self.v_max = rospy.get_param("/parameter_scheduling_mpc/max_linear_vel")
        self.omega_max = rospy.get_param("/parameter_scheduling_mpc/max_angular_vel")
        self.wheel_rad = rospy.get_param("/parameter_scheduling_mpc/wheel_radius")
        self.wheel_separation = rospy.get_param("/parameter_scheduling_mpc/wheel_separation")
        self.RK4 = rospy.get_param("/parameter_scheduling_mpc/RK4")
        self.cmd_vel_topic = rospy.get_param("/parameter_scheduling_mpc/cmd_vel_topic")
        self.base_frame = rospy.get_param("/parameter_scheduling_mpc/base_frame")
        self.xy_tol = rospy.get_param("/parameter_scheduling_mpc/xy_tol")
        self.yaw_tol = 1 - np.cos(rospy.get_param("/parameter_scheduling_mpc/yaw_tol"))
        self.cost_r = rospy.get_param("/parameter_scheduling_mpc/cost_r")
        self.cost_alpha = rospy.get_param("/parameter_scheduling_mpc/cost_alpha")
        self.cost_phi = rospy.get_param("/parameter_scheduling_mpc/cost_phi")
        self.warm_start = rospy.get_param("/parameter_scheduling_mpc/warm_start")
        self.mode = rospy.get_param("/parameter_scheduling_mpc/mode")
        self.minimize_rot_err = rospy.get_param("/parameter_scheduling_mpc/minimize_rot_err")
        self.publish_path = rospy.get_param("/parameter_scheduling_mpc/publish_path")
        self.switching = rospy.get_param("/parameter_scheduling_mpc/switching")
        self.mdp_data = rospy.get_param("/parameter_scheduling_mpc/mdp_data")
        self.target_sub = rospy.Subscriber(
            "/move_base_simple/goal",
            PoseStamped,
            self.targetCallback
        )
        self.tfListener = TransformListener()
        if self.use_odom:
            self.odom_sub = rospy.Subscriber(
                "/odom",
                Odometry,
                self.odomCallback
            )
        self.gx = self.gy = self.gyaw = None

        # declare velocity publisher
        self.vel_msg = Twist()
        self.vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=10)
        self.path_pub = rospy.Publisher("/parameter_scheduling_mpc/local_plan", Path, queue_size=10)

        # MDP setup
        if self.mdp_data!="fixed":
            mdp = MarkovDecisionProcess()
            mdp.load(self.mdp_data)
            self.policy = mdp.policy

        # MPC setup
        self.setup_MPC()


    def spin(self):
        rate = rospy.Rate(self.loop_rate)
        while not rospy.is_shutdown():
            if self.tic is not None:
                self.run()
            rate.sleep()


    def pose_from_tf(self):

        try:
            position, quaternion = self.tfListener.lookupTransform(
                "/map",
                self.base_frame,
                self.tfListener.getLatestCommonTime("/map", self.base_frame)
            )
            self.curx = position[0]
            self.cury = position[1]
            (roll, pitch, yaw) = euler_from_quaternion(quaternion)
            self.curyaw = yaw
        except rospy.ROSException:
            rospy.loginfo(
                "Warn: %s - No transform from [%s] to [map]."%(self.base_frame, self.base_frame)
            )


    def pose_from_msg(self, msg):

        self.curx = msg.pose.pose.position.x
        self.cury = msg.pose.pose.position.y
        quat = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        (roll, pitch, yaw) = euler_from_quaternion(quat)
        self.curyaw = yaw


    def odomCallback(self, msg):

        self.pose_from_msg(msg)
        

    def targetCallback(self, msg):

        self.gx = msg.pose.position.x
        self.gy = msg.pose.position.y
        quat = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        )
        (roll, pitch, yaw) = euler_from_quaternion(quat)
        self.gyaw = yaw

        self.activate_mpc = True

        if self.tic is not None:
            print('Recieved a new target pose. previous target pose will be ignored.\n')
        print('Target pose :', self.gx, self.gy, self.gyaw)
        self.tic = rospy.get_time()


    def run(self):

        if not self.use_odom:
            # Robot pose from /tf if use_odom is False
            self.pose_from_tf()

        if self.is_reached():
            vel = [0.0, 0.0]
            self.u_prev = np.zeros((2))
            self.toc = rospy.get_time()
            print('Time elapsed:', self.toc - self.tic, '\n')
            self.tic = None
        else:
            if not controller.isInitialized:
                self.setup_MPC()
            u = self.iterate_MPC()
            self.u_prev = u.copy()
            vel = [
                self.wheel_rad * (u[0] + u[1]) / 2,
                self.wheel_rad * (u[1] - u[0]) / self.wheel_separation
            ]

        self.vel_msg.linear.x  = vel[0]
        self.vel_msg.angular.z = vel[1]
        self.vel_pub.publish(self.vel_msg)


    def is_reached(self):
        if type(self.gyaw)==type(None):
            return True
        else:
            yaw_diff = 1 - np.cos(self.gyaw - self.curyaw)
            xy_diff = np.linalg.norm(
                (self.gx - self.curx, self.gy - self.cury)
            )
            return yaw_diff < self.yaw_tol and xy_diff < self.xy_tol


    def setup_MPC(self):
        if self.switching==False:
            if self.mode=='polar':
                self.mpc = NmpcPolar(
                    T=self.T,
                    N=self.N,
                    R_u=self.regularization_u*np.eye(2),
                    R_vel=np.diag(
                      np.array([self.regularization_v, self.regularization_omega])
                    ),
                    w_max=self.w_max,
                    a_max=self.a_max,
                    v_max=self.v_max,
                    omega_max=self.omega_max,
                    vel_constraint='quadratic',
                    max_iter=100,
                    ode='RK4' if self.RK4 else 'forward euler',
                    wheel_rad=self.wheel_rad,
                    wheel_separation=self.wheel_separation
                )
            else:
                self.mpc = NmpcCartesian(
                    T=self.T,
                    N=self.N,
                    R_u=self.regularization_u*np.eye(2),
                    R_vel=np.diag(
                      np.array([self.regularization_v, self.regularization_omega])
                    ),
                    w_max=self.w_max,
                    a_max=self.a_max,
                    v_max=self.v_max,
                    omega_max=self.omega_max,
                    vel_constraint='quadratic',
                    max_iter=100,
                    ode='RK4' if self.RK4 else 'forward euler',
                    wheel_rad=self.wheel_rad,
                    wheel_separation=self.wheel_separation
                )
        else:
            self.mpc = NmpcSwitching(
                T=self.T,
                N=self.N,
                R_u=self.regularization_u*np.eye(2),
                R_vel=np.diag(
                  np.array([self.regularization_v, self.regularization_omega])
                ),
                w_max=self.w_max,
                a_max=self.a_max,
                v_max=self.v_max,
                omega_max=self.omega_max,
                vel_constraint='quadratic',
                max_iter=100,
                ode='RK4' if self.RK4 else 'forward euler',
                r_tol=self.xy_tol,
                mode=self.mode,
                wheel_rad=self.wheel_rad,
                wheel_separation=self.wheel_separation
            )
        self.u_prev = np.zeros((2))
        self.isInitialized = True


    def iterate_MPC(self):

        if not self.use_odom:
            self.pose_from_tf() # Robot pose from /tf if use_odom is False

        err_x = self.curx - self.gx
        err_y = self.cury - self.gy
        x_init = np.array(
            [
                np.cos(self.gyaw)*err_x + np.sin(self.gyaw)*err_y,
                -np.sin(self.gyaw)*err_x + np.cos(self.gyaw)*err_y,
                normalize_angle(self.curyaw - self.gyaw)
            ]
        )
        
        r = np.linalg.norm(x_init[:2])
        psi = normalize_angle(-x_init[2])
        alpha = normalize_angle(np.arctan2(x_init[1],x_init[0]) + psi - np.pi)

        # get NMPC cost function parameter from policy
        if self.mdp_data=="fixed":
            if self.minimize_rot_err:
                theta = np.array([self.cost_r, self.cost_alpha, self.cost_phi])
            else:
                theta = np.array([self.cost_r, self.cost_alpha, 0])
        else:
            if self.minimize_rot_err:
                theta = self.policy.get_action([r, alpha, psi])
            else:
                theta = self.policy.get_action([r, alpha])

        X, U = self.mpc.solve([r, alpha, psi], theta, u_prev=self.u_prev, return_trajectory=True)
        if self.publish_path:
            path_msg = Path()
            path_msg.header.frame_id = "map"
            path_msg.header.stamp = rospy.Time.now()
            for point in np.reshape(X,[-1,3]):
                temp = PoseStamped()
                temp.header = path_msg.header
                xr = point[0] * np.cos(np.pi + point[1] - point[2]),
                yr = point[0] * np.sin(np.pi + point[1] - point[2]),
                temp.pose.position.x = xr * np.cos(self.gyaw) - yr * np.sin(self.gyaw) + self.gx
                temp.pose.position.y = xr * np.sin(self.gyaw) + yr * np.cos(self.gyaw) + self.gy
                quat = quaternion_from_euler(0.0, 0.0, self.gyaw - point[2])
                temp.pose.orientation.z = quat[2]
                temp.pose.orientation.w = quat[3]
                path_msg.poses.append(temp)
            self.path_pub.publish(path_msg)

        return np.array(
            U[0, :]
        )



if __name__ == '__main__':

    try:
        controller = ParameterSchedulingMPC()
        tic = time.time()
        while not controller.isInitialized:
            toc = time()
            if toc - tic > 10:
                rospy.loginfo("MPC initialization timeout (10s).")
                break
        if controller.isInitialized:
            controller.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("node terminated.")
