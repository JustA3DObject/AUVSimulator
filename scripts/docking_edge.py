import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu, NavSatFix, FluidPressure
from geometry_msgs.msg import TwistWithCovarianceStamped
from datetime import datetime
import math
import os
import numpy as np
import pandas as pd
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

# Configuration Parameters

# Sensor Noise Configuration (standard deviations for simulated noise)
NOISE_POS_STD   = 0.15      # Position noise standard deviation [m]
NOISE_DEPTH_STD = 0.05      # Depth noise standard deviation [m]
NOISE_YAW_STD   = 0.05      # Yaw noise standard deviation [rad]
NOISE_DOCK_STD  = 0.2       # Dock position noise standard deviation [m]

# GNC Parameters for heading control
KP_HEADING      = -4.5      # Proportional gain for heading control
KI_HEADING      = -0.1      # Integral gain for heading control
KD_HEADING      = -8.0      # Derivative gain for heading control (Damping)
KP_SWAY         = 50.0      # Proportional gain for sway (cross‑track) control
INTEGRAL_LIMIT  = 15.0      # Anti‑windup limit for heading integral term

# Depth Control Parameters
KP_DEPTH        = 80.0      # Proportional gain for depth control
KI_DEPTH        = 5.0       # Integral gain for depth control
DEPTH_I_LIMIT   = 20.0      # Anti‑windup limit for depth integral

# Pure Pursuit Parameters (final docking phase)
DOCKING_TOLERANCE = 2.5     # Distance [m] to consider the AUV docked

# Artificial Potential Field Phase Parameters
STAGING_DIST    = 20.0      # Distance from dock mouth to staging point [m]
APF_K_ATT       = 1.0       # Attractive force gain to staging point
APF_K_REP       = 100.0     # Repulsive force gain (to avoid dock)
APF_K_TAN       = 300.0     # Tangential force gain (creates clockwise rotation)
APF_R_INF       = 15.0      # Influence radius of the dock obstacle [m]

# Coordinates of four lights on the dock (in dock body frame)
LIGHT_COORDS = [
    [0.0, 4.53528, -2.58534],        # Light 1
    [2.585345, 4.53528, 0.0],        # Light 2
    [0.0, 4.53528, 2.585345],        # Light 3
    [-2.585345, 4.53528, 0.0]        # Light 4
]

# Utility Functions

def get_yaw(q):
    """
    Extract yaw angle from a quaternion.
    :param q: quaternion (with attributes w, x, y, z)
    :return: yaw angle in radians
    """
    t3 = +2.0 * (q.w * q.z + q.x * q.y)
    t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(t3, t4)


def normalize_angle(angle):
    """
    Normalize an angle to the range [-pi, pi].
    :param angle: angle in radians
    :return: normalized angle
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quaternion_matrix(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.
    :param q: quaternion (with attributes w, x, y, z)
    :return: 3x3 numpy array representing the rotation
    """
    w, x, y, z = q.w, q.x, q.y, q.z
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])


# Filtering Classes

class SimpleEKF:
    """
    Extended Kalman Filter for AUV position (x, y, depth) and yaw.
    Uses a constant‑velocity model for prediction and direct measurements for update.
    """
    def __init__(self, q_pos, q_yaw, r_pos, r_depth, r_yaw):
        """
        :param q_pos: process noise variance for position
        :param q_yaw: process noise variance for yaw
        :param r_pos: measurement noise std for position (x, y)
        :param r_depth: measurement noise std for depth
        :param r_yaw: measurement noise std for yaw
        """
        self.x = np.zeros((4, 1))          # State: [x, y, depth, yaw]
        self.P = np.eye(4)                 # State covariance
        self.Q = np.diag([q_pos, q_pos, q_pos, q_yaw])   # Process noise covariance
        self.R = np.diag([r_pos**2, r_pos**2, r_depth**2, r_yaw**2])  # Measurement noise covariance
        self.H = np.eye(4)                  # Measurement matrix (direct observation)

    def predict(self, dt, v):
        """
        Prediction step using constant‑velocity model.
        :param dt: time step [s]
        :param v: forward speed [m/s] (body‑fixed x‑velocity)
        """
        yaw = self.x[3, 0]
        # Update state: x += v*cos(yaw)*dt, y += v*sin(yaw)*dt
        self.x[0, 0] += v * math.cos(yaw) * dt
        self.x[1, 0] += v * math.sin(yaw) * dt
        # Jacobian of the motion model
        F = np.eye(4)
        F[0, 3] = -v * math.sin(yaw) * dt
        F[1, 3] = v * math.cos(yaw) * dt
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """
        Update step with a measurement vector.
        :param z: measurement vector [x, y, depth, yaw]
        :return: updated state vector
        """
        y = z - (self.H @ self.x)
        y[3, 0] = normalize_angle(y[3, 0])   # Normalize yaw innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.x[3, 0] = normalize_angle(self.x[3, 0])
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x


class StaticKF:
    """
    Kalman filter for a static position (e.g., dock pose).
    Assumes the object is stationary (Q small) and filters noisy measurements.
    """
    def __init__(self, r_std):
        """
        :param r_std: measurement noise standard deviation
        """
        self.x = np.zeros((3, 1))           # State: [x, y, z]
        self.P = np.eye(3) * 10.0           # Initial covariance
        self.Q = np.eye(3) * 0.0001         # Process noise (small, since stationary)
        self.R = np.eye(3) * (r_std**2)     # Measurement noise covariance
        self.H = np.eye(3)                  # Measurement matrix
        self.initialized = False

    def update(self, z):
        """
        Update the filter with a new measurement.
        :param z: measurement vector [x, y, z]
        :return: filtered state
        """
        z = np.array(z).reshape((3, 1))
        if not self.initialized:
            self.x = z
            self.initialized = True
            return self.x.flatten()
        # Prediction (just add process noise)
        self.P = self.P + self.Q
        # Update
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(3) - K @ self.H) @ self.P
        return self.x.flatten()


# Main GNC Node
class DockingGNC(Node):
    """
    ROS2 node for autonomous docking.
    """
    def __init__(self):
        super().__init__('auv_docking_gnc')

        # State variables
        self.dock_pose = None                 
        self.auv_x = 0.0                      
        self.auv_y = 0.0                      
        self.auv_z = 0.0                      
        self.auv_yaw = 0.0                    
        self.auv_forward_speed = 0.0          
        self.s_d_prev = None                  
        self.prev_yaw_error_deg = None     

        # Mission phase tracking
        self.reached_staging_point = False     

        # Telemetry dictionary for dashboard
        self.telemetry = {
            'phase': 'APF Initial', 'dist': 0.0, 'z_err': 0.0, 'yaw_err': 0.0,
            'xte': 0.0, 'speed': 0.0, 'rd': 0.0, 'plato_u': 0.0
        }

        # Dock geometry
        self.light_center_local = np.mean(LIGHT_COORDS, axis=0)

        # Integral terms for heading and depth control
        self.integral_error = 0.0
        self.depth_integral = 0.0

        # Time keeping for control loop
        self.prev_time = self.get_clock().now()
        self.last_odom_time = None

        # Filters
        self.auv_ekf = SimpleEKF(q_pos=0.01, q_yaw=0.01,
                                 r_pos=NOISE_POS_STD,
                                 r_depth=NOISE_DEPTH_STD,
                                 r_yaw=NOISE_YAW_STD)
        self.dock_kf = StaticKF(r_std=NOISE_DOCK_STD)

        # Publishers 
        self.pub_port = self.create_publisher(Float64, '/girona500/ThrusterSurgePort/setpoint', 1)
        self.pub_stbd = self.create_publisher(Float64, '/girona500/ThrusterSurgeStarboard/setpoint', 1)
        self.pub_sway = self.create_publisher(Float64, '/girona500/ThrusterSway/setpoint', 1)
        self.pub_heave_b = self.create_publisher(Float64, '/girona500/ThrusterHeaveBow/setpoint', 1)
        self.pub_heave_s = self.create_publisher(Float64, '/girona500/ThrusterHeaveStern/setpoint', 1)

        # Subscribers
        self.create_subscription(Odometry, '/girona500/dynamics/odometry', self.auv_cb, 1)
        self.create_subscription(Odometry, '/dock/position', self.dock_cb, 1)

        # Data logger
        self.data_logger = AUVLogger()
        self.add_logger_subscribers()

        # Control timer
        self.timer = self.create_timer(0.02, self.control_loop)
        self.get_logger().info("Docking Node Started.")

    def add_logger_subscribers(self):
        """
        Add subscriptions for all logger topics.
        Each subscription writes the received data to the logger.
        """
        self.create_subscription(Float64, '/girona500/ThrusterSurgePort/setpoint',
                                 lambda m: self.data_logger.log_val('ThrusterSurgePort', m.data), 1)
        self.create_subscription(Float64, '/girona500/ThrusterSurgeStarboard/setpoint',
                                 lambda m: self.data_logger.log_val('ThrusterSurgeStarboard', m.data), 1)
        self.create_subscription(Float64, '/girona500/ThrusterSway',
                                 lambda m: self.data_logger.log_val('ThrusterSway', m.data), 1)
        self.create_subscription(Float64, '/girona500/ThrusterHeaveBow/setpoint',
                                 lambda m: self.data_logger.log_val('ThrusterHeaveBow', m.data), 1)
        self.create_subscription(Float64, '/girona500/ThrusterHeaveStern/setpoint',
                                 lambda m: self.data_logger.log_val('ThrusterHeaveStern', m.data), 1)
        self.create_subscription(Odometry, '/girona500/dynamics/odometry',
                                 self.data_logger.log_odometry, 1)
        self.create_subscription(Imu, '/girona500/imu',
                                 self.data_logger.log_imu, 1)
        self.create_subscription(Image, '/girona500/camera/image_raw',
                                 self.data_logger.log_camera, 1)
        self.create_subscription(Image, '/girona500/fls/image_raw',
                                 self.data_logger.log_fls, 1)

    # Callbacks
    def dock_cb(self, msg):
        """
        Callback for dock position. Applies noise and Kalman filtering.
        :param msg: Odometry message containing dock pose
        """
        self.dock_pose = msg.pose.pose
        # Add noise
        noisy_x = self.dock_pose.position.x + np.random.normal(0, NOISE_DOCK_STD)
        noisy_y = self.dock_pose.position.y + np.random.normal(0, NOISE_DOCK_STD)
        noisy_z = self.dock_pose.position.z + np.random.normal(0, NOISE_DOCK_STD)
        # Filter
        filtered_dock = self.dock_kf.update([noisy_x, noisy_y, noisy_z])
        self.dock_pose.position.x = filtered_dock[0]
        self.dock_pose.position.y = filtered_dock[1]
        self.dock_pose.position.z = filtered_dock[2]

    def auv_cb(self, msg):
        """
        Callback for AUV odometry. Adds noise and runs EKF.
        :param msg: Odometry message with AUV pose and twist
        """
        # Extract clean values from simulation
        clean_x = msg.pose.pose.position.x
        clean_y = msg.pose.pose.position.y
        clean_z = msg.pose.pose.position.z
        clean_yaw = get_yaw(msg.pose.pose.orientation)

        # Add sensor noise
        noisy_x = clean_x + np.random.normal(0, NOISE_POS_STD)
        noisy_y = clean_y + np.random.normal(0, NOISE_POS_STD)
        noisy_z = clean_z + np.random.normal(0, NOISE_DEPTH_STD)
        noisy_yaw = normalize_angle(clean_yaw + np.random.normal(0, NOISE_YAW_STD))

        # Forward speed (body‑fixed x‑velocity) – used in prediction
        self.auv_forward_speed = msg.twist.twist.linear.x

        current_time = self.get_clock().now()
        if self.last_odom_time is None:
            # First measurement: initialize EKF state
            self.last_odom_time = current_time
            self.auv_ekf.x = np.array([[noisy_x], [noisy_y], [noisy_z], [noisy_yaw]])
            self.auv_x, self.auv_y, self.auv_z, self.auv_yaw = noisy_x, noisy_y, noisy_z, noisy_yaw
            return

        dt = (current_time - self.last_odom_time).nanoseconds / 1e9
        self.last_odom_time = current_time

        if dt > 0:
            # Predict step
            self.auv_ekf.predict(dt, self.auv_forward_speed)
            # Update step with measurement
            z_meas = np.array([[noisy_x], [noisy_y], [noisy_z], [noisy_yaw]])
            filtered_state = self.auv_ekf.update(z_meas)

            # Store filtered state
            self.auv_x = float(filtered_state[0, 0])
            self.auv_y = float(filtered_state[1, 0])
            self.auv_z = float(filtered_state[2, 0])
            self.auv_yaw = float(filtered_state[3, 0])

    # Thruster interface
    def set_thrust(self, port, stbd, sway, heave):
        """
        Publish thruster setpoints after saturation.
        """
        port = max(min(port, 200.0), -200.0)
        stbd = max(min(stbd, 200.0), -200.0)
        sway = max(min(sway, 200.0), -200.0)
        heave = max(min(heave, 200.0), -200.0)

        self.pub_port.publish(Float64(data=float(port)))
        self.pub_stbd.publish(Float64(data=float(stbd)))
        self.pub_sway.publish(Float64(data=float(sway)))
        self.pub_heave_b.publish(Float64(data=float(heave)))
        self.pub_heave_s.publish(Float64(data=float(heave)))

    # Main control loop
    def control_loop(self):
        """
        Timer callback. Computes guidance and control outputs.
        """
        if self.dock_pose is None:
            return

        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds / 1e9
        if dt <= 0:
            return
        self.prev_time = current_time

        # Compute important 3D points
        P_tail_3d = np.array([self.dock_pose.position.x, self.dock_pose.position.y, self.dock_pose.position.z])
        P_tail_2d = P_tail_3d[:2]
        dock_z_target = self.dock_pose.position.z
        R_dock = quaternion_matrix(self.dock_pose.orientation)

        offset_rotated = R_dock.dot(self.light_center_local)
        P_mouth_3d = P_tail_3d + offset_rotated
        P_mouth_2d = P_mouth_3d[:2]

        P_auv_3d = np.array([self.auv_x, self.auv_y, self.auv_z])
        P_auv_2d = P_auv_3d[:2]

        # Depth control
        depth_error = dock_z_target - self.auv_z
        self.depth_integral += depth_error * dt
        self.depth_integral = max(min(self.depth_integral, DEPTH_I_LIMIT), -DEPTH_I_LIMIT)
        u_heave = -1.0 * ((KP_DEPTH * depth_error) + (KI_DEPTH * self.depth_integral))

        # Staging point computation
        len_mouth = np.linalg.norm(offset_rotated)
        if len_mouth > 0.001:
            u_mouth_3d = offset_rotated / len_mouth
        else:
            u_mouth_3d = np.array([0.0, 1.0, 0.0])

        P_staging_3d = P_mouth_3d + (u_mouth_3d * STAGING_DIST)
        P_staging_2d = P_staging_3d[:2]

        dist_to_staging = np.linalg.norm(P_staging_2d - P_auv_2d)
        dist_to_dock = np.linalg.norm(P_tail_3d - P_auv_3d)

        # Phase selection
        if not self.reached_staging_point:
            if dist_to_staging < 2.0 and abs(depth_error) < 1.0:
                self.reached_staging_point = True
                
                # RESET MEMORY FOR PLATO PHASE
                self.integral_error = 0.0
                self.prev_yaw_error_deg = None
                
                self.get_logger().info("Staging Point Reached! Deactivating APF.")
            else:
                v_att = P_staging_2d - P_auv_2d
                F_att = APF_K_ATT * (v_att / max(dist_to_staging, 0.001))

                P_obs_2d = (P_tail_2d + P_mouth_2d) / 2.0
                v_obs = P_auv_2d - P_obs_2d
                d_obs = np.linalg.norm(v_obs)

                F_rep = np.array([0.0, 0.0])
                F_tan = np.array([0.0, 0.0])

                if d_obs < APF_R_INF:
                    NO_FLY_RADIUS = 5.0
                    safe_d_boundary = max(d_obs - NO_FLY_RADIUS, 0.001)
                    true_d = max(d_obs, 0.1)
                    n_vec = v_obs / true_d                     
                    t_vec = np.array([n_vec[1], -n_vec[0]])    
                    
                    mag = (1.0 / safe_d_boundary) - (1.0 / (APF_R_INF - NO_FLY_RADIUS))
                    mag = max(mag, 0.0)                        

                    F_rep = APF_K_REP * (mag**2) * n_vec
                    F_tan = APF_K_TAN * (mag**2) * t_vec

                F_total = F_att + F_rep + F_tan
                desired_yaw = math.atan2(F_total[1], F_total[0])

                yaw_error = normalize_angle(desired_yaw - self.auv_yaw)
                yaw_error_deg = math.degrees(yaw_error)

                # Heading derivative calculation
                if self.prev_yaw_error_deg is None:
                    self.prev_yaw_error_deg = yaw_error_deg

                # Normalize the angular difference to prevent derivative spikes
                raw_diff = yaw_error_deg - self.prev_yaw_error_deg
                delta_yaw = math.degrees(normalize_angle(math.radians(raw_diff)))

                yaw_error_dot = delta_yaw / dt
                self.prev_yaw_error_deg = yaw_error_deg

                # Heading integral
                self.integral_error += yaw_error * dt
                self.integral_error = max(min(self.integral_error, INTEGRAL_LIMIT), -INTEGRAL_LIMIT)

                # PID Yaw thrust 
                raw_u_yaw = (KP_HEADING * yaw_error_deg) + (KI_HEADING * self.integral_error) + (KD_HEADING * yaw_error_dot)
                u_yaw = max(min(raw_u_yaw, 50.0), -50.0)

                u_sway = 0.0

                if abs(yaw_error_deg) > 60.0:
                    u_surge = -40.0
                else:
                    u_surge = -90.0 * (1.0 - (abs(yaw_error_deg) / 90.0))

                u_port = u_surge + u_yaw
                u_stbd = u_surge - u_yaw
                self.set_thrust(u_port, u_stbd, u_sway, u_heave)

                self.telemetry.update({
                    'phase': 'APF Staging', 'dist': dist_to_dock, 'z_err': depth_error,
                    'yaw_err': yaw_error_deg, 'xte': 0.0, 'speed': self.auv_forward_speed,
                    'rd': 0.0, 'plato_u': 0.0
                })
                
                self.get_logger().info(
                    f"[APF Phase] Dist2Waypoint: {dist_to_staging:.2f}m | Dist: {dist_to_dock:.2f}m | "
                    f"Z-Err: {depth_error:.2f}m | YawErr: {yaw_error_deg:.1f}° | XTE: 0.00m | "
                    f"FwdSpeed: {self.auv_forward_speed:.2f}m/s | Rd: 0.00m | PLATO_u: 0.000",
                    throttle_duration_sec=0.2
                )
                return

        # PLATO + Pure Pursuit Phase
        vec_path = P_tail_2d - P_mouth_2d
        len_path = np.linalg.norm(vec_path)
        if len_path < 0.001:
            return   

        u_path = vec_path / len_path          
        vec_MA = P_auv_2d - P_mouth_2d        
        s_curr = np.dot(vec_MA, u_path)       
        P_closest = P_mouth_2d + (s_curr * u_path)

        STRICT_LOOKAHEAD = 6.0 
        P_goal = P_closest + (STRICT_LOOKAHEAD * u_path)

        if dist_to_dock < DOCKING_TOLERANCE:
            self.get_logger().info("DOCKED! Stopping Thrusters.")
            self.set_thrust(0.0, 0.0, 0.0, 0.0)
            return

        vec_to_goal = P_goal - P_auv_2d
        desired_yaw = math.atan2(vec_to_goal[1], vec_to_goal[0])
        yaw_error = normalize_angle(desired_yaw - self.auv_yaw)
        yaw_error_deg = math.degrees(yaw_error)

        # Heading derivative calculation
        if self.prev_yaw_error_deg is None:
            self.prev_yaw_error_deg = yaw_error_deg
            
        # Normalize the angular difference to prevent derivative spikes
        raw_diff = yaw_error_deg - self.prev_yaw_error_deg
        delta_yaw = math.degrees(normalize_angle(math.radians(raw_diff)))
        
        yaw_error_dot = delta_yaw / dt
        self.prev_yaw_error_deg = yaw_error_deg

        self.integral_error += yaw_error * dt
        self.integral_error = max(min(self.integral_error, INTEGRAL_LIMIT), -INTEGRAL_LIMIT)

        # NEW: Full PID Yaw thrust 
        raw_u_yaw = (KP_HEADING * yaw_error_deg) + (KI_HEADING * self.integral_error) + (KD_HEADING * yaw_error_dot)
        u_yaw = max(min(raw_u_yaw, 50.0), -50.0)

        vec_err_world = P_closest - P_auv_2d
        c = math.cos(self.auv_yaw)
        s = math.sin(self.auv_yaw)
        body_sway_error = (-s * vec_err_world[0]) + (c * vec_err_world[1])
        u_sway = KP_SWAY * body_sway_error

        # PLATO Speed regulation
        s_min, s_c, a_max = 0.1, 1.0, 0.2
        n_min, n_max = 3.0, 6.0
        r_d, f_dec, K_ps = 4.0, 7.0, 10.0

        s_t = abs(self.auv_forward_speed)      
        d_t = dist_to_dock                     

        R_d_t = r_d + f_dec * (s_t**2) / (2.0 * a_max)
        u_t = 0.0   

        if d_t <= R_d_t:
            n_t = n_min + (n_max - n_min) * min(1.0, s_t / s_c)
            safe_d_t = max(d_t, 0.001)
            log_num = math.log(safe_d_t) - math.log(r_d)
            log_den = math.log(R_d_t) - math.log(r_d)
            s_d_t = s_min + (s_c - s_min) * (max(0.0, log_num / log_den) ** n_t)

            if self.s_d_prev is None:
                s_d_dot = 0.0
            else:
                s_d_dot = (s_d_t - self.s_d_prev) / dt
            self.s_d_prev = s_d_t

            e_s = s_d_t - s_t
            u_t = (K_ps * e_s) + s_d_dot
            u_surge = (-140.0 * (s_t / s_c)) + (-160.0 * u_t)
        else:
            self.s_d_prev = None
            if abs(yaw_error_deg) > 45.0:
                u_surge = 0.0
            else:
                u_surge = -140.0 * (1.0 - (abs(yaw_error_deg) / 45.0))

        u_port = u_surge + u_yaw
        u_stbd = u_surge - u_yaw
        self.set_thrust(u_port, u_stbd, u_sway, u_heave)

        self.telemetry.update({
            'phase': 'PLATO Docking', 'dist': dist_to_dock, 'z_err': depth_error,
            'yaw_err': yaw_error_deg, 'xte': body_sway_error, 'speed': self.auv_forward_speed,
            'rd': R_d_t, 'plato_u': u_t
        })

        self.get_logger().info(
            f"[PLATO Phase] Dist2Waypoint: {dist_to_staging:.2f}m | Dist: {dist_to_dock:.2f}m | "
            f"Z-Err: {depth_error:.2f}m | YawErr: {yaw_error_deg:.1f}° | XTE: {body_sway_error:.2f}m | "
            f"FwdSpeed: {self.auv_forward_speed:.2f}m/s | Rd: {R_d_t:.2f}m | PLATO_u: {u_t:.3f}",
            throttle_duration_sec=0.2
        )

# Data Logger Node
class AUVLogger(Node):
    """
    ROS2 node for logging sensor and actuator topics to a Parquet file.
    Data is buffered and written in batches.
    """
    def __init__(self):
        super().__init__('auv_data_logger')
        self.lock = threading.Lock()

        # Create output folder
        self.folder_name = "AUVSimData"
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name, exist_ok=True)

        # Timestamped filename
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = os.path.join(self.folder_name, f"AUV_docking_{timestamp_str}.parquet")

        # Data buffer
        self.data_buffer = []
        self.buffer_limit = 200

        # Subscribers for additional sensors
        self.sub_gps = self.create_subscription(NavSatFix, '/girona500/gps', self.log_gps, 1)
        self.sub_pre = self.create_subscription(FluidPressure, '/girona500/pressure', self.log_pressure, 1)
        self.sub_dvl = self.create_subscription(TwistWithCovarianceStamped, '/girona500/dvl_twist', self.log_dvl, 1)
        self.sub_acc = self.create_subscription(Imu, '/girona500/accelerometer', self.log_accel, 1)

    def write_line(self, topic, data):
        """
        Append a log entry to the buffer. Flush if buffer limit reached.
        :param topic: string identifier for the data source
        :param data: data to store (will be converted to string)
        """
        with self.lock:
            timestamp = self.get_clock().now().to_msg().sec + (self.get_clock().now().to_msg().nanosec / 1e9)
            self.data_buffer.append({'timestamp': timestamp, 'topic': topic, 'data': str(data)})
            if len(self.data_buffer) >= self.buffer_limit:
                self.flush_to_disk_internal()

    def flush_to_disk_internal(self):
        """
        Write buffered data to Parquet file (called under lock).
        """
        if not self.data_buffer:
            return
        try:
            df = pd.DataFrame(self.data_buffer)
            if not os.path.isfile(self.filename):
                df.to_parquet(self.filename, engine='pyarrow', index=False)
            else:
                existing_df = pd.read_parquet(self.filename, engine='pyarrow')
                pd.concat([existing_df, df], ignore_index=True).to_parquet(self.filename, engine='pyarrow', index=False)
            self.data_buffer = []
        except Exception as e:
            self.get_logger().error(f"Log Flush Error: {e}")

    # Logging callbacks
    def log_val(self, name, val):
        """Log a single value."""
        self.write_line(name, val)

    def log_odometry(self, msg):
        """Log odometry pose."""
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self.write_line("Odometry_Pose", [p.x, p.y, p.z, o.w, o.x, o.y, o.z])

    def log_imu(self, msg):
        """Log IMU orientation."""
        o = msg.orientation
        self.write_line("IMU", [o.x, o.y, o.z, o.w])

    def log_gps(self, msg):
        """Log GPS fix."""
        self.write_line("GPS", [msg.latitude, msg.longitude])

    def log_pressure(self, msg):
        """Log pressure reading."""
        self.write_line("Pressure", msg.fluid_pressure)

    def log_dvl(self, msg):
        """Log DVL twist (velocity + covariance)."""
        v = msg.twist.twist.linear
        cov = list(msg.twist.covariance)
        self.write_line("DVL", [v.x, v.y, v.z, cov])

    def log_accel(self, msg):
        """Log accelerometer reading."""
        a = msg.linear_acceleration if hasattr(msg, 'linear_acceleration') else msg.accel.linear
        self.write_line("Accelerometer", [a.x, a.y, a.z])

    def log_camera(self, msg):
        """Log camera image info (dimensions only)."""
        self.write_line("Main Camera", f"{msg.width}x{msg.height}")

    def log_fls(self, msg):
        """Log forward‑looking sonar image info."""
        self.write_line("FLS Sonar", f"{msg.width}x{msg.height}")

    def close(self):
        """Flush any remaining data before shutdown."""
        with self.lock:
            self.flush_to_disk_internal()


# Main
def main(args=None):
    rclpy.init(args=args)

    # Create nodes
    gnc_node = DockingGNC()
    logger_node = gnc_node.data_logger

    # Use multi‑threaded executor to run both nodes concurrently
    executor = MultiThreadedExecutor()
    executor.add_node(gnc_node)
    executor.add_node(logger_node)

    # Start ROS spinning in a background thread
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    # Dashboard
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2, width_ratios=[3, 1])

    ax_map = fig.add_subplot(gs[0], projection='3d')
    ax_dash = fig.add_subplot(gs[1])

    ax_map.set_title("Live AUV 3D Trajectory")
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.set_zlabel("Depth (m)")
    ax_map.invert_zaxis()
    ax_map.grid(True)

    # Plot elements
    path_line, = ax_map.plot([], [], [], 'b-', alpha=0.5, label='AUV Path')
    auv_dot, = ax_map.plot([], [], [], 'bo', markersize=8, label='AUV Position')
    dock_dot, = ax_map.plot([], [], [], 'rX', markersize=10, label='Dock')
    target_line, = ax_map.plot([], [], [], 'g--', alpha=0.8, label='LOS Target Vector')
    ax_map.legend(loc='upper right')
    ax_dash.set_title("Instrument Cluster")
    ax_dash.set_aspect('equal')
    ax_dash.axis('off')
    theta_arc = np.linspace(np.pi, 0, 100)
    ax_dash.plot(np.cos(theta_arc), np.sin(theta_arc), 'k-', lw=2)

    max_gauge_speed = 1.5
    for val in np.linspace(0, max_gauge_speed, 7):
        angle = np.pi - (val / max_gauge_speed) * np.pi
        xi, yi = np.cos(angle), np.sin(angle)
        ax_dash.plot([0.9*xi, xi], [0.9*yi, yi], 'k-', lw=1.5)
        ax_dash.text(1.1*xi, 1.1*yi, f"{val:.1f}", ha='center', va='center', fontsize=9)

    speed_needle, = ax_dash.plot([0, 0], [0, 1], 'r-', lw=3)
    speed_text = ax_dash.text(0, -0.2, "0.00 m/s", ha='center', fontsize=14, fontweight='bold', color='blue')

    telemetry_box = ax_dash.text(0, -0.5, "", ha='center', va='top', fontsize=11, family='monospace',
                                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

    ax_dash.set_xlim(-1.3, 1.3)
    ax_dash.set_ylim(-1.5, 1.3)

    # History for trajectory
    x_history = []
    y_history = []
    z_history = []

    def update_plot(frame):
        """
        Animation update function: refreshes 3D plot and dashboard.
        """
        if gnc_node.dock_pose is None:
            return path_line, auv_dot, dock_dot, target_line, speed_needle, speed_text, telemetry_box

        curr_x = gnc_node.auv_x
        curr_y = gnc_node.auv_y
        curr_z = gnc_node.auv_z
        dock_x = gnc_node.dock_pose.position.x
        dock_y = gnc_node.dock_pose.position.y
        dock_z = gnc_node.dock_pose.position.z

        if curr_x != 0.0 and curr_y != 0.0:
            x_history.append(curr_x)
            y_history.append(curr_y)
            z_history.append(curr_z)

            # Update 3D trajectory
            path_line.set_data(x_history, y_history)
            path_line.set_3d_properties(z_history)

            auv_dot.set_data([curr_x], [curr_y])
            auv_dot.set_3d_properties([curr_z])

            dock_dot.set_data([dock_x], [dock_y])
            dock_dot.set_3d_properties([dock_z])

            # Compute and draw lookahead target line
            P_tail_2d = np.array([dock_x, dock_y])
            R_dock = quaternion_matrix(gnc_node.dock_pose.orientation)
            offset_rotated = R_dock.dot(gnc_node.light_center_local)
            P_mouth_2d = P_tail_2d + offset_rotated[:2]

            vec_path = P_tail_2d - P_mouth_2d
            len_path = np.linalg.norm(vec_path)
            if len_path > 0.001:
                u_path = vec_path / len_path
                P_auv_2d = np.array([curr_x, curr_y])
                vec_MA = P_auv_2d - P_mouth_2d
                s_curr = np.dot(vec_MA, u_path)
                P_closest = P_mouth_2d + (s_curr * u_path)

                STRICT_LOOKAHEAD = 3.0
                P_goal = P_closest + (STRICT_LOOKAHEAD * u_path)

                target_line.set_data([curr_x, P_goal[0]], [curr_y, P_goal[1]])
                target_line.set_3d_properties([curr_z, dock_z])

            # Auto‑adjust axes
            all_x = x_history + [dock_x]
            all_y = y_history + [dock_y]
            all_z = z_history + [dock_z]
            margin = 5.0
            ax_map.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax_map.set_ylim(min(all_y) - margin, max(all_y) + margin)
            z_min, z_max = min(all_z), max(all_z)
            if abs(z_max - z_min) < 1.0:
                ax_map.set_zlim(z_max + 2.0, z_min - 2.0)
            else:
                ax_map.set_zlim(z_max + margin, z_min - margin)

            # Dashboard updates
            t = gnc_node.telemetry
            current_speed = abs(t['speed'])
            clamped_speed = min(current_speed, max_gauge_speed)
            needle_angle = np.pi - (clamped_speed / max_gauge_speed) * np.pi
            speed_needle.set_data([0, 0.85 * np.cos(needle_angle)], [0, 0.85 * np.sin(needle_angle)])
            speed_text.set_text(f"{current_speed:.2f} m/s")

            dash_str = (
                f"Dist to Dock : {t['dist']:6.2f} m\n"
                f"Z-Error      : {t['z_err']:6.2f} m\n"
                f"Yaw-Error    : {t['yaw_err']:6.1f} °\n"
                f"Cross-Track  : {t['xte']:6.2f} m\n"
                f"Decel Rad(Rd): {t['rd']:6.2f} m\n"
                f"PLATO Accel U: {t['plato_u']:6.3f}  "
            )
            telemetry_box.set_text(dash_str)

        return path_line, auv_dot, dock_dot, target_line, speed_needle, speed_text, telemetry_box

    ani = animation.FuncAnimation(fig, update_plot, interval=100, cache_frame_data=False)

    try:
        plt.show()   # Blocks until window is closed
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down plotting and stopping AUV Thrusters...")
        gnc_node.set_thrust(0.0, 0.0, 0.0, 0.0)
        logger_node.close()
        rclpy.shutdown()


if __name__ == '__main__':
    main()