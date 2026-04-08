import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64, Float64MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu, NavSatFix, FluidPressure
from geometry_msgs.msg import TwistWithCovarianceStamped
from datetime import datetime
import math
import os
import numpy as np
import pandas as pd
import threading
import cv2
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# Configuration Parameters
# Sensor Noise Configuration (standard deviations for simulated noise)
NOISE_POS_STD   = 0.15      # Position noise standard deviation [m]
NOISE_DEPTH_STD = 0.05      # Depth noise standard deviation [m]
NOISE_YAW_STD   = 0.05      # Yaw noise standard deviation [rad]
NOISE_DOCK_STD  = 0.2       # Dock position noise standard deviation [m]

# GNC Parameters for heading control
KP_HEADING      = -12.0     # Proportional gain for heading control
KI_HEADING      = -0.8      # Integral gain for heading control
KD_HEADING      = -15.0     # Derivative gain for heading control (Damping)
KP_SWAY         = 50.0      # Proportional gain for sway (cross-track) control
INTEGRAL_LIMIT  = 15.0      # Anti-windup limit for heading integral term

# Depth & Pitch Control Parameters
KP_DEPTH        = 80.0      # Proportional gain for depth control
KI_DEPTH        = 5.0       # Integral gain for depth control
DEPTH_I_LIMIT   = 20.0      # Anti-windup limit for depth integral
KP_PITCH_VIS    = 10.0      # Proportional gain for visual pitch control
KD_PITCH_VIS    = 5.0       # Derivative gain for visual pitch control

# Pure Pursuit Parameters (final docking phase)
DOCKING_TOLERANCE = 2.5     # Distance [m] to consider the AUV docked

# Artificial Potential Field Phase Parameters
STAGING_DIST    = 20.0      # Distance from dock mouth to staging point [m]
APF_K_ATT       = 1.0       # Attractive force gain to staging point
APF_K_REP       = 100.0     # Repulsive force gain (to avoid dock)
APF_K_TAN       = 300.0     # Tangential force gain (creates clockwise rotation)
APF_R_INF       = 15.0      # Influence radius of the dock obstacle [m]

# CBF Parameters (Speed - PLATO)
CBF_S_MIN       = 0.05      # Minimum safe speed bound
CBF_S_MAX       = 1.5       # Maximum safe speed bound
CBF_R_D         = 2.0       # Critical deceleration radius
CBF_U_MIN       = -1.5      # Minimum allowable acceleration
CBF_U_MAX       = 1.5       # Maximum allowable acceleration

CBF_ALPHA_1     = 2.0       # Class K function parameter 1
CBF_ALPHA_2     = 2.0       # Class K function parameter 2
CBF_GAMMA_1     = 1.0       # CBF constraint relaxation 1
CBF_GAMMA_2     = 1.0       # CBF constraint relaxation 2

# CBF Parameters (Guidance - Relative Degree 2 HOCBF)
CBF_Z_TOL       = 1.5       # Depth safety tolerance [m]
CBF_XTE_TOL     = 2.0       # Cross-track error safety tolerance [m]
CBF_YAW_TOL_RAD = math.radians(60.0) # Yaw error safety tolerance [rad]
CBF_PITCH_TOL_RAD = math.radians(20.0) # Pitch error safety tolerance [rad]

CBF_GAMMA1_Z    = 1.5; CBF_GAMMA2_Z    = 1.5
CBF_GAMMA1_XTE  = 1.5; CBF_GAMMA2_XTE  = 1.5
CBF_GAMMA1_YAW  = 1.5; CBF_GAMMA2_YAW  = 1.5
CBF_GAMMA1_PITCH= 1.5; CBF_GAMMA2_PITCH= 1.5

ACT_SCALE_Z     = 50.0      # Actuator scaling for heave constraint mapping
ACT_SCALE_XTE   = 50.0      # Actuator scaling for sway constraint mapping
ACT_SCALE_YAW   = 100.0     # Actuator scaling for yaw constraint mapping
ACT_SCALE_PITCH = 80.0      # Actuator scaling for pitch constraint mapping

# Coordinates of four lights on the dock (in dock body frame)
LIGHT_COORDS = [
    [0.0, 4.53528, -2.58534],        
    [2.585345, 4.53528, 0.0],        
    [0.0, 4.53528, 2.585345],        
    [-2.585345, 4.53528, 0.0]        
]

# Utility Functions
def get_yaw(q):
    """
    Extracts the yaw angle from a quaternion.
    
    :param q: geometry_msgs.msg.Quaternion object containing w, x, y, z components.
    :return: Yaw angle in radians.
    """
    t3 = +2.0 * (q.w * q.z + q.x * q.y)
    t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(t3, t4)

def get_pitch(q):
    """
    Extracts the pitch angle from a quaternion.
    
    :param q: geometry_msgs.msg.Quaternion object containing w, x, y, z components.
    :return: Pitch angle in radians.
    """
    t2 = +2.0 * (q.w * q.y - q.z * q.x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    return math.asin(t2)

def normalize_angle(angle):
    """
    Normalizes an angle to be within the range [-pi, pi].
    
    :param angle: The angle to normalize, in radians.
    :return: The normalized angle in radians.
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

def quaternion_matrix(q):
    """
    Converts a quaternion representation into a 3x3 rotation matrix.
    
    :param q: geometry_msgs.msg.Quaternion object containing w, x, y, z components.
    :return: A 3x3 numpy array representing the rotation matrix.
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
    A simple Extended Kalman Filter (EKF) implementation for 4-DOF AUV state estimation.
    Tracks X position, Y position, depth (Z), and yaw orientation.
    """
    def __init__(self, q_pos, q_yaw, r_pos, r_depth, r_yaw):
        """
        Initializes the EKF with process and measurement noise covariances.
        
        :param q_pos: Process noise variance for positional states (x, y, z).
        :param q_yaw: Process noise variance for the yaw state.
        :param r_pos: Measurement noise standard deviation for positional states (x, y).
        :param r_depth: Measurement noise standard deviation for the depth state (z).
        :param r_yaw: Measurement noise standard deviation for the yaw state.
        """
        self.x = np.zeros((4, 1))          
        self.P = np.eye(4)                 
        self.Q = np.diag([q_pos, q_pos, q_pos, q_yaw])   
        self.R = np.diag([r_pos**2, r_pos**2, r_depth**2, r_yaw**2])  
        self.H = np.eye(4)

    def predict(self, dt, v):
        """
        Executes the prediction step of the EKF using a constant-velocity kinematic model.
        
        :param dt: Time delta since the last update in seconds.
        :param v: The forward linear velocity (surge speed) of the AUV.
        """
        # Propagate state using non-linear kinematics (constant body-velocity model)
        yaw = self.x[3, 0]
        self.x[0, 0] += v * math.cos(yaw) * dt
        self.x[1, 0] += v * math.sin(yaw) * dt
        
        # Calculate the Jacobian of the state transition matrix
        F = np.eye(4)
        F[0, 3] = -v * math.sin(yaw) * dt
        F[1, 3] = v * math.cos(yaw) * dt
        
        # Propagate covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """
        Executes the update step of the EKF using new sensor measurements.
        
        :param z: A 4x1 numpy array containing the latest measurements [x, y, z, yaw]^T.
        :return: The updated 4x1 state vector.
        """
        # Calculate innovation (measurement residual)
        y = z - (self.H @ self.x)
        y[3, 0] = normalize_angle(y[3, 0])  # Prevent yaw innovation wrap-around errors 
        
        # Compute Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state and covariance estimates
        self.x = self.x + (K @ y)
        self.x[3, 0] = normalize_angle(self.x[3, 0])
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x

class StaticKF:
    """
    A linear Kalman Filter designed specifically for smoothing stationary 3D targets,
    such as the fixed position of the docking station.
    """
    def __init__(self, r_std):
        """
        Initializes the Static Kalman Filter.
        
        :param r_std: Measurement noise standard deviation for the 3D position coordinates.
        """
        self.x = np.zeros((3, 1))           
        self.P = np.eye(3) * 10.0           
        self.Q = np.eye(3) * 0.0001         # Minimal process noise assuming a static target
        self.R = np.eye(3) * (r_std**2)     
        self.H = np.eye(3)                  
        self.initialized = False

    def update(self, z):
        """
        Updates the filter with a new 3D position measurement of the static target.
        
        :param z: A list or array containing the measured [x, y, z] coordinates.
        :return: A flattened 1D numpy array representing the filtered [x, y, z] coordinates.
        """
        z = np.array(z).reshape((3, 1))
        
        # Handle first measurement directly to avoid slow convergence
        if not self.initialized:
            self.x = z
            self.initialized = True
            return self.x.flatten()
            
        # Prediction step
        self.P = self.P + self.Q
        
        # Update step
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(3) - K @ self.H) @ self.P
        
        return self.x.flatten()

# Vision Node
class VisionNode(Node):
    """
    ROS2 Node responsible for processing incoming camera images, detecting
    docking station lights using HSV color masking, and publishing tracked features.
    """
    def __init__(self):
        """
        Initializes the VisionNode, setting up publishers, subscribers, and CvBridge.
        """
        super().__init__('auv_vision')
        self.bridge = CvBridge()
        
        self.sub_cam = self.create_subscription(
            Image, 
            '/girona500/front_camera/image_raw/image_color', 
            self.image_cb, 
            qos_profile_sensor_data
        )
            
        self.pub_cam = self.create_publisher(Image, '/girona500/front_camera/tracked_lights', 10)
        self.pub_vision_features = self.create_publisher(Float64MultiArray, '/vision/dock_features', 10)

    def image_cb(self, msg):
        """
        Callback triggered on receiving a new camera image. Applies color thresholds
        to detect dock lights, identifies patterns, and publishes the tracked target coordinates.
        
        :param msg: sensor_msgs.msg.Image message from the forward-facing camera.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        # Convert to HSV for robust color thresholding against lighting variations
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Isolate the central blue light (Primary target)
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Isolate the outer green lights (Secondary orientation pattern)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mode = 0.0
        tracked_cx, tracked_cy = 0.0, 0.0

        # Process green contours to find potential light sources
        green_centers = []
        for c in contours_green:
            if cv2.contourArea(c) > 2:  # Filter out tiny noise artifacts
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                if radius > 1:
                    green_centers.append((float(x), float(y)))
                    cv2.circle(cv_image, (int(x), int(y)), int(radius) + 3, (0, 255, 0), 2)

        is_circular_pattern = False
        geom_cx, geom_cy = 0.0, 0.0

        # If 4 or more green lights are detected, verify if they form the known dock pattern
        if len(green_centers) >= 4:
            # Calculate geometric centroid of all detected green lights
            geom_cx = sum(pt[0] for pt in green_centers) / len(green_centers)
            geom_cy = sum(pt[1] for pt in green_centers) / len(green_centers)
            
            # Verify circularity by calculating the variance of distances from the center
            distances = [math.hypot(pt[0]-geom_cx, pt[1]-geom_cy) for pt in green_centers]
            avg_dist = sum(distances) / len(distances)
            
            if avg_dist > 10.0:  # Ensure the pattern isn't just a cluster of noise
                variance = sum((d - avg_dist)**2 for d in distances) / len(distances)
                std_dev = math.sqrt(variance)
                
                # A low standard deviation means the points form a relatively perfect circle
                if std_dev < (0.25 * avg_dist):
                    is_circular_pattern = True
                    cv2.circle(cv_image, (int(geom_cx), int(geom_cy)), int(avg_dist), (0, 255, 255), 2)
                    cv2.putText(cv_image, "CIRCULAR PATTERN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Process blue contours to extract the primary tracking coordinate
        blue_center = None
        if contours_blue:
            largest_blue = max(contours_blue, key=cv2.contourArea)
            if cv2.contourArea(largest_blue) > 2:
                ((x, y), radius) = cv2.minEnclosingCircle(largest_blue)
                blue_center = (float(x), float(y))
                cv2.circle(cv_image, (int(x), int(y)), int(radius) + 5, (255, 0, 0), 3)

        # Logic hierarchy for defining the visual tracking mode
        if blue_center:
            tracked_cx, tracked_cy = blue_center
            if is_circular_pattern:
                # Best Case: We see the center AND the surrounding pattern.
                mode = 2.0  
            else:
                # Acceptable Case: We only see the center blue light clearly.
                mode = 1.0  
        elif is_circular_pattern:
            # Fallback Case: Blue light lost, but geometric center of green pattern is valid.
            tracked_cx = geom_cx
            tracked_cy = geom_cy
            mode = 3.0  

        # Annotate and publish standard metrics if a target is acquired
        if mode > 0.0:
            cv_image = cv2.drawMarker(cv_image, (int(tracked_cx), int(tracked_cy)), (0, 0, 255), 
                           markerType=cv2.MARKER_CROSS, markerSize=30, thickness=3)
            
            text = "FULL LOCK" if mode == 2.0 else "CENTER" if mode == 1.0 else "FALLBACK"
            cv2.putText(cv_image, text, (int(tracked_cx) + 10, int(tracked_cy) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            vis_msg = Float64MultiArray()
            vis_msg.data = [tracked_cx, tracked_cy, 0.0, 0.0, mode]
            self.pub_vision_features.publish(vis_msg)
        else:
            # Publish zeroed array to notify controllers that tracking is lost
            vis_msg = Float64MultiArray()
            vis_msg.data = [0.0, 0.0, 0.0, 0.0, 0.0]
            self.pub_vision_features.publish(vis_msg)

        try:
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.pub_cam.publish(out_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Publish Error: {e}")

# Main GNC Node
class DockingGNC(Node):
    """
    The core Guidance, Navigation, and Control (GNC) node for the AUV. Manages
    mission phases (APF staging to PLATO docking), incorporates HOCBF safety filters,
    and publishes final thruster allocations.
    """
    def __init__(self):
        """
        Initializes the DockingGNC node, state variables, telemetric buffers,
        Kalman filters, publishers for thrusters, and core data subscriptions.
        """
        super().__init__('auv_docking_gnc')

        self.dock_pose = None                 
        self.auv_x = 0.0                      
        self.auv_y = 0.0                      
        self.auv_z = 0.0                      
        self.auv_yaw = 0.0
        self.auv_pitch = 0.0
        self.auv_forward_speed = 0.0          
        self.s_d_prev = None                  
        self.reached_staging_point = False

        # State trackers and filters for PID/HOCBF
        self.prev_yaw_error_deg = None     
        self.prev_pitch_err = None
        self.prev_depth_err = None
        self.prev_sway_err = None
        self.prev_yaw_error_rad = None
        self.prev_pitch_error_rad = None
        
        self.filtered_depth_error_dot = 0.0
        self.filtered_sway_error_dot = 0.0
        self.filtered_yaw_error_dot_rad = 0.0
        self.filtered_pitch_error_dot_rad = 0.0
        self.filtered_pitch_dot = 0.0

        self.vis_cx = 680.0
        self.vis_cy = 512.0
        self.vis_valid = False
        self.vis_mode = 0.0
        self.prev_vis_mode = 0.0
        self.last_vis_time = self.get_clock().now()

        self.telemetry = {
            'phase': 'APF Initial', 'dist': 0.0, 'z_err': 0.0, 'yaw_err': 0.0,
            'xte': 0.0, 'speed': 0.0, 'rd': 0.0, 'plato_u': 0.0, 'vis_align': 'WAITING',
            'alpha': 0.0            
        }

        self.light_center_local = np.mean(LIGHT_COORDS, axis=0)

        self.integral_error = 0.0
        self.depth_integral = 0.0
        self.sway_integral = 0.0
        self.pitch_integral = 0.0

        self.prev_time = self.get_clock().now()
        self.last_odom_time = None

        self.auv_ekf = SimpleEKF(q_pos=0.01, q_yaw=0.01,
                                 r_pos=NOISE_POS_STD,
                                 r_depth=NOISE_DEPTH_STD,
                                 r_yaw=NOISE_YAW_STD)
        self.dock_kf = StaticKF(r_std=NOISE_DOCK_STD)

        self.pub_port = self.create_publisher(Float64, '/girona500/ThrusterSurgePort/setpoint', 1)
        self.pub_stbd = self.create_publisher(Float64, '/girona500/ThrusterSurgeStarboard/setpoint', 1)
        self.pub_sway = self.create_publisher(Float64, '/girona500/ThrusterSway/setpoint', 1)
        self.pub_heave_b = self.create_publisher(Float64, '/girona500/ThrusterHeaveBow/setpoint', 1)
        self.pub_heave_s = self.create_publisher(Float64, '/girona500/ThrusterHeaveStern/setpoint', 1)

        self.create_subscription(Odometry, '/girona500/dynamics/odometry', self.auv_cb, 1)
        self.create_subscription(Odometry, '/dock/position', self.dock_cb, 1)
        self.create_subscription(Float64MultiArray, '/vision/dock_features', self.vision_cb, 1)

        self.data_logger = AUVLogger()
        self.add_logger_subscribers()

        self.timer = self.create_timer(0.02, self.control_loop)
        self.get_logger().info("Docking Node Started. CBF + Vision Enabled.")

    def vision_cb(self, msg):
        """
        Callback to process incoming visual features derived by the VisionNode.
        
        :param msg: std_msgs.msg.Float64MultiArray containing track coordinates and lock mode.
        """
        self.vis_mode = msg.data[4]
        if self.vis_mode > 0.0:
            self.vis_cx = msg.data[0]
            self.vis_cy = msg.data[1]
            self.vis_valid = True
            self.last_vis_time = self.get_clock().now()

    def add_logger_subscribers(self):
        """
        Initializes the appropriate subscriptions required by the localized AUVLogger instance
        to record thruster states, telemetry, and camera metadata.
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

    def dock_cb(self, msg):
        """
        Callback handling incoming odometry measurements of the docking station.
        Applies synthetic noise and feeds it into the StaticKF for state smoothing.
        
        :param msg: nav_msgs.msg.Odometry message of the dock pose.
        """
        self.dock_pose = msg.pose.pose
        noisy_x = self.dock_pose.position.x + np.random.normal(0, NOISE_DOCK_STD)
        noisy_y = self.dock_pose.position.y + np.random.normal(0, NOISE_DOCK_STD)
        noisy_z = self.dock_pose.position.z + np.random.normal(0, NOISE_DOCK_STD)
        
        filtered_dock = self.dock_kf.update([noisy_x, noisy_y, noisy_z])
        
        self.dock_pose.position.x = filtered_dock[0]
        self.dock_pose.position.y = filtered_dock[1]
        self.dock_pose.position.z = filtered_dock[2]

    def auv_cb(self, msg):
        """
        Callback processing the AUV's dynamic odometry. Applies synthetic positional
        and orientation noise, driving the SimpleEKF for state prediction and updating.
        
        :param msg: nav_msgs.msg.Odometry message of the AUV pose and twist.
        """
        # Extract ground truth from the simulation message
        clean_x = msg.pose.pose.position.x
        clean_y = msg.pose.pose.position.y
        clean_z = msg.pose.pose.position.z
        clean_yaw = get_yaw(msg.pose.pose.orientation)
        clean_pitch = get_pitch(msg.pose.pose.orientation)

        # Inject white Gaussian noise to simulate real-world sensor uncertainties
        noisy_x = clean_x + np.random.normal(0, NOISE_POS_STD)
        noisy_y = clean_y + np.random.normal(0, NOISE_POS_STD)
        noisy_z = clean_z + np.random.normal(0, NOISE_DEPTH_STD)
        noisy_yaw = normalize_angle(clean_yaw + np.random.normal(0, NOISE_YAW_STD))

        self.auv_pitch = clean_pitch + np.random.normal(0, NOISE_YAW_STD)
        self.auv_forward_speed = msg.twist.twist.linear.x

        current_time = self.get_clock().now()
        
        # Initialize filter on the first received message
        if self.last_odom_time is None:
            self.last_odom_time = current_time
            self.auv_ekf.x = np.array([[noisy_x], [noisy_y], [noisy_z], [noisy_yaw]])
            self.auv_x, self.auv_y, self.auv_z, self.auv_yaw = noisy_x, noisy_y, noisy_z, noisy_yaw
            return

        dt = (current_time - self.last_odom_time).nanoseconds / 1e9
        self.last_odom_time = current_time

        # Run EKF iterations for smoothed state estimation
        if dt > 0:
            self.auv_ekf.predict(dt, self.auv_forward_speed)
            z_meas = np.array([[noisy_x], [noisy_y], [noisy_z], [noisy_yaw]])
            filtered_state = self.auv_ekf.update(z_meas)

            self.auv_x = float(filtered_state[0, 0])
            self.auv_y = float(filtered_state[1, 0])
            self.auv_z = float(filtered_state[2, 0])
            self.auv_yaw = float(filtered_state[3, 0])

    def set_thrust(self, port, stbd, sway, heave_b, heave_s):
        """
        Clamps and publishes force setpoints to the 5 degrees-of-freedom thrusters.
        
        :param port: Thrust value for the surge-port thruster.
        :param stbd: Thrust value for the surge-starboard thruster.
        :param sway: Thrust value for the lateral sway thruster.
        :param heave_b: Thrust value for the bow heave thruster.
        :param heave_s: Thrust value for the stern heave thruster.
        """
        # Ensure motor limits are not exceeded
        port = max(min(port, 200.0), -200.0)
        stbd = max(min(stbd, 200.0), -200.0)
        sway = max(min(sway, 200.0), -200.0)
        heave_b = max(min(heave_b, 200.0), -200.0)
        heave_s = max(min(heave_s, 200.0), -200.0)

        self.pub_port.publish(Float64(data=float(port)))
        self.pub_stbd.publish(Float64(data=float(stbd)))
        self.pub_sway.publish(Float64(data=float(sway)))
        self.pub_heave_b.publish(Float64(data=float(heave_b)))
        self.pub_heave_s.publish(Float64(data=float(heave_s)))

    def control_loop(self):
        """
        The primary 50Hz control timer callback. Calculates errors based on current 
        AUV states against the dock pose, switches execution between APF Staging 
        and PLATO Terminal Docking, applies Control Barrier Functions (CBFs), and 
        issues thruster commands.
        """
        if self.dock_pose is None:
            return

        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds / 1e9
        if dt <= 0:
            return

        self.prev_time = current_time

        # Transform dock coordinates to find the virtual staging area and light centers
        P_tail_3d = np.array([self.dock_pose.position.x, self.dock_pose.position.y, self.dock_pose.position.z])
        P_tail_2d = P_tail_3d[:2]
        dock_z_target = self.dock_pose.position.z
        R_dock = quaternion_matrix(self.dock_pose.orientation)

        offset_rotated = R_dock.dot(self.light_center_local)
        P_mouth_3d = P_tail_3d + offset_rotated
        P_mouth_2d = P_mouth_3d[:2]

        P_auv_3d = np.array([self.auv_x, self.auv_y, self.auv_z])
        P_auv_2d = P_auv_3d[:2]

        depth_error = dock_z_target - self.auv_z

        # Calculate a unit vector pointing directly out of the dock mouth
        len_mouth = np.linalg.norm(offset_rotated)
        if len_mouth > 0.001:
            u_mouth_3d = offset_rotated / len_mouth
        else:
            u_mouth_3d = np.array([0.0, 1.0, 0.0])

        # Project the staging point linearly out from the dock
        P_staging_3d = P_mouth_3d + (u_mouth_3d * STAGING_DIST)
        P_staging_2d = P_staging_3d[:2]

        dist_to_staging = np.linalg.norm(P_staging_2d - P_auv_2d)
        dist_to_dock = np.linalg.norm(P_tail_3d - P_auv_3d)

        # PHASE 1: APF Staging
        if not self.reached_staging_point:
            # Check condition to transition smoothly into Phase 2
            if dist_to_staging < 2.0 and abs(depth_error) < 1.0:
                self.reached_staging_point = True
                self.integral_error = 0.0
                
                # Reset error states for HOCBF transitions to prevent integration windup spikes
                self.prev_yaw_error_deg = None
                self.prev_yaw_error_rad = None
                self.prev_pitch_err = None
                self.prev_pitch_error_rad = None
                self.prev_depth_err = None
                self.prev_sway_err = None
                
                self.filtered_pitch_dot = 0.0
                self.filtered_depth_error_dot = 0.0
                self.filtered_sway_error_dot = 0.0
                self.filtered_yaw_error_dot_rad = 0.0
                self.filtered_pitch_error_dot_rad = 0.0
                
                self.get_logger().info("Staging Point Reached! Deactivating APF. Engaging PLATO + Visual + CBF.")
            else:
                # Maintain depth utilizing a standard PI controller
                self.depth_integral += depth_error * dt
                self.depth_integral = max(min(self.depth_integral, DEPTH_I_LIMIT), -DEPTH_I_LIMIT)
                u_heave = -1.0 * ((KP_DEPTH * depth_error) + (KI_DEPTH * self.depth_integral))

                # APF Pitch control (Level stabilization)
                apf_pitch_err_deg = 0.0 - math.degrees(self.auv_pitch) 
                
                if self.prev_pitch_err is None:
                    self.prev_pitch_err = apf_pitch_err_deg
                    self.filtered_pitch_dot = 0.0
                    
                raw_pitch_dot = (apf_pitch_err_deg - self.prev_pitch_err) / dt
                self.prev_pitch_err = apf_pitch_err_deg
                
                # Low-pass filter the derivative term to prevent thruster chatter
                self.filtered_pitch_dot = (0.7 * self.filtered_pitch_dot) + (0.3 * raw_pitch_dot)
                
                KP_PITCH_APF = 5.0 
                KD_PITCH_APF = 2.0  
                u_pitch = (KP_PITCH_APF * apf_pitch_err_deg) + (KD_PITCH_APF * self.filtered_pitch_dot)
                u_pitch = max(min(u_pitch, 40.0), -40.0)
                
                u_heave_b = u_heave + u_pitch
                u_heave_s = u_heave - u_pitch
                
                # APF Attractive Force: Pulls AUV towards the safe staging point
                v_att = P_staging_2d - P_auv_2d
                F_att = APF_K_ATT * (v_att / max(dist_to_staging, 0.001))

                P_obs_2d = (P_tail_2d + P_mouth_2d) / 2.0
                v_obs = P_auv_2d - P_obs_2d
                d_obs = np.linalg.norm(v_obs)

                F_rep = np.array([0.0, 0.0])
                F_tan = np.array([0.0, 0.0])

                # APF Obstacle Avoidance: Triggers only if AUV is within influence radius of dock
                if d_obs < APF_R_INF:
                    NO_FLY_RADIUS = 5.0
                    safe_d_boundary = max(d_obs - NO_FLY_RADIUS, 0.001)
                    true_d = max(d_obs, 0.1)
                    n_vec = v_obs / true_d                     
                    t_vec = np.array([n_vec[1], -n_vec[0]])    
                    
                    mag = (1.0 / safe_d_boundary) - (1.0 / (APF_R_INF - NO_FLY_RADIUS))
                    mag = max(mag, 0.0)                        

                    # Repulsive force pushes AUV away from the physical dock obstacle
                    F_rep = APF_K_REP * (mag**2) * n_vec
                    # Tangential force generates a rotational vector to navigate around the dock
                    F_tan = APF_K_TAN * (mag**2) * t_vec

                # Vector summation yields the desired heading direction
                F_total = F_att + F_rep + F_tan
                desired_yaw = math.atan2(F_total[1], F_total[0])

                yaw_error = normalize_angle(desired_yaw - self.auv_yaw)
                yaw_error_deg = math.degrees(yaw_error)

                if self.prev_yaw_error_deg is None:
                    self.prev_yaw_error_deg = yaw_error_deg

                raw_diff = yaw_error_deg - self.prev_yaw_error_deg
                delta_yaw = math.degrees(normalize_angle(math.radians(raw_diff)))

                yaw_error_dot = delta_yaw / dt
                self.prev_yaw_error_deg = yaw_error_deg

                yaw_error_rad = math.radians(yaw_error_deg)
                self.integral_error += yaw_error_rad * dt
                self.integral_error = max(min(self.integral_error, INTEGRAL_LIMIT), -INTEGRAL_LIMIT)

                # PID calculation for yaw thrust allocation
                raw_u_yaw = (KP_HEADING * yaw_error_deg) + (KI_HEADING * self.integral_error) + (KD_HEADING * yaw_error_dot)
                u_yaw = max(min(raw_u_yaw, 50.0), -50.0)

                u_sway = 0.0

                # Adaptive surge velocity depending on heading alignment error
                if abs(yaw_error_deg) > 60.0:
                    u_surge = -30.0
                else:
                    u_surge = -50.0 * (1.0 - (abs(yaw_error_deg) / 90.0))

                u_port = u_surge + u_yaw
                u_stbd = u_surge - u_yaw
                
                self.set_thrust(u_port, u_stbd, u_sway, u_heave_b, u_heave_s)

                apf_pitch_err_deg = -math.degrees(self.auv_pitch) 
                self.telemetry.update({
                    'phase': 'APF Staging', 'dist': dist_to_dock, 'z_err': depth_error,
                    'yaw_err': yaw_error_deg, 'xte': 0.0, 'speed': self.auv_forward_speed,
                    'rd': 0.0, 'plato_u': 0.0, 'vis_align': 'WAITING'
                })

                self.get_logger().info(
                    f"[APF Phase] Dist2Waypoint: {dist_to_staging:.2f}m | Dist: {dist_to_dock:.2f}m | "
                    f"Z-Err: {depth_error:.2f}m | YawErr: {yaw_error_deg:.1f}° | PitchErr: {apf_pitch_err_deg:.1f}° | "
                    f"XTE: 0.00m | FwdSpeed: {self.auv_forward_speed:.2f}m/s",
                    throttle_duration_sec=0.2
                )
                return

        # PHASE 2: PTerminal Docking
        vec_path = P_tail_2d - P_mouth_2d
        len_path = np.linalg.norm(vec_path)
        if len_path < 0.001:
            return   

        # Pure Pursuit: Compute orthogonal projection of AUV onto the optimal docking path
        u_path = vec_path / len_path          
        vec_MA = P_auv_2d - P_mouth_2d        
        s_curr = np.dot(vec_MA, u_path)       
        P_closest = P_mouth_2d + (s_curr * u_path)

        # Place the lookahead target down the path
        STRICT_LOOKAHEAD = 10.0 
        P_goal = P_closest + (STRICT_LOOKAHEAD * u_path)

        # Termination criterion for docking sequence
        if dist_to_dock < DOCKING_TOLERANCE:
            self.get_logger().info("DOCKED! Stopping Thrusters.")
            self.set_thrust(0.0, 0.0, 0.0, 0.0, 0.0)
            return

        # Compute Default Guidance Errors using geometry
        vec_to_goal = P_goal - P_auv_2d
        desired_yaw = math.atan2(vec_to_goal[1], vec_to_goal[0])
        default_yaw_err = math.degrees(normalize_angle(desired_yaw - self.auv_yaw))
        default_pitch_err = -math.degrees(self.auv_pitch)

        # Visual Timeout mechanism to prevent using stale tracking data
        if (current_time - self.last_vis_time).nanoseconds / 1e9 > 0.5:
            self.vis_valid = False

        # Compute Blending Weight α: Linearly interpolate reliance on camera vs. EKF over distance
        d = min(dist_to_dock, 20.0)
        alpha = 0.5 + 0.5 * (1.0 - d / 20.0)   # α=0.5 at 20m, α=1.0 at 0m
        if not self.vis_valid:
            alpha = 0.0
        # Ignore vision entirely at close range to prevent light saturation tracking failure
        if dist_to_dock < 5.5:
            alpha = 0.0
        self.telemetry['alpha'] = alpha

        # Camera Errors (if valid and alpha>0)
        if self.vis_valid and alpha > 0.0:
            # Map pixel delta from center to degrees using horizontal/vertical FOV constants
            deg_per_px_x = 80.0 / 1360.0
            deg_per_px_y = 60.0 / 1024.0
            cam_yaw_err = (self.vis_cx - 680.0) * deg_per_px_x
            cam_pitch_err = -(self.vis_cy - 512.0) * deg_per_px_y

            # Sensor fusion: blend visually calculated error with geometric odometry error
            yaw_error_deg = (alpha * cam_yaw_err) + ((1.0 - alpha) * default_yaw_err)
            pitch_error_deg = (alpha * cam_pitch_err) + ((1.0 - alpha) * default_pitch_err)

            if self.vis_mode == 2.0:
                self.telemetry['vis_align'] = "FULL LOCK"
            elif self.vis_mode == 1.0:
                self.telemetry['vis_align'] = "CENTER"
            else:
                self.telemetry['vis_align'] = "FALLBACK"
        else:
            yaw_error_deg = default_yaw_err
            pitch_error_deg = default_pitch_err
            self.telemetry['vis_align'] = "EKF (LOST)" if not self.vis_valid else "NO VIS"

        # Radiometric conversions for the CBF Filters
        yaw_error_rad = math.radians(yaw_error_deg)
        pitch_error_rad = math.radians(pitch_error_deg)

        # Calculate Cross-track (Sway) Error relative to the body frame
        vec_err_world = P_closest - P_auv_2d
        c = math.cos(self.auv_yaw)
        s = math.sin(self.auv_yaw)
        body_sway_error = (-s * vec_err_world[0]) + (c * vec_err_world[1])

        # Nominal PID Controllers
        # HEAVE Nominal Control
        self.depth_integral += depth_error * dt
        self.depth_integral = max(min(self.depth_integral, DEPTH_I_LIMIT), -DEPTH_I_LIMIT)
        u_heave_nom = -1.0 * ((KP_DEPTH * depth_error) + (KI_DEPTH * self.depth_integral))

        # PITCH Nominal Control
        KP_PITCH = 6.0; KD_PITCH = 2.5; KI_PITCH = 1.0; PITCH_I_LIMIT = 8.0  
        if self.prev_pitch_err is None:
            self.prev_pitch_err = pitch_error_deg
            self.filtered_pitch_dot = 0.0
            self.pitch_integral = 0.0

        raw_pitch_dot_deg = (pitch_error_deg - self.prev_pitch_err) / dt
        self.prev_pitch_err = pitch_error_deg
        self.filtered_pitch_dot = (0.7 * self.filtered_pitch_dot) + (0.3 * raw_pitch_dot_deg)

        self.pitch_integral += pitch_error_deg * dt
        self.pitch_integral = max(min(self.pitch_integral, PITCH_I_LIMIT), -PITCH_I_LIMIT)
        u_pitch_nom = (KP_PITCH * pitch_error_deg) + (KI_PITCH * self.pitch_integral) + (KD_PITCH * self.filtered_pitch_dot)

        # SWAY Nominal Control
        KP_SWAY = 180.0; KI_SWAY = 30.0; SWAY_I_LIMIT = 15.0
        self.sway_integral += body_sway_error * dt
        self.sway_integral = max(min(self.sway_integral, SWAY_I_LIMIT), -SWAY_I_LIMIT)
        u_sway_nom = (KP_SWAY * body_sway_error) + (KI_SWAY * self.sway_integral)

        # YAW Nominal Control
        if self.prev_yaw_error_deg is None:
            self.prev_yaw_error_deg = yaw_error_deg

        raw_diff = yaw_error_deg - self.prev_yaw_error_deg
        delta_yaw = math.degrees(normalize_angle(math.radians(raw_diff)))
        yaw_error_dot_deg = delta_yaw / dt
        self.prev_yaw_error_deg = yaw_error_deg

        self.integral_error += yaw_error_rad * dt
        self.integral_error = max(min(self.integral_error, INTEGRAL_LIMIT), -INTEGRAL_LIMIT)
        u_yaw_nom = (KP_HEADING * yaw_error_deg) + (KI_HEADING * self.integral_error) + (KD_HEADING * yaw_error_dot_deg)

        # Calculate Error Rates for HOCBF (With Low-Pass Smoothing)
        alpha_f = 0.2  

        # Depth
        if self.prev_depth_err is None: self.prev_depth_err = depth_error
        depth_error_dot_raw = (depth_error - self.prev_depth_err) / dt
        self.filtered_depth_error_dot = (1.0 - alpha_f) * self.filtered_depth_error_dot + alpha_f * depth_error_dot_raw
        self.prev_depth_err = depth_error
        depth_error_dot = self.filtered_depth_error_dot

        # Sway
        if self.prev_sway_err is None: self.prev_sway_err = body_sway_error
        sway_error_dot_raw = (body_sway_error - self.prev_sway_err) / dt
        self.filtered_sway_error_dot = (1.0 - alpha_f) * self.filtered_sway_error_dot + alpha_f * sway_error_dot_raw
        self.prev_sway_err = body_sway_error
        sway_error_dot = self.filtered_sway_error_dot

        # Yaw
        if self.prev_yaw_error_rad is None: self.prev_yaw_error_rad = yaw_error_rad
        yaw_error_dot_raw_rad = normalize_angle(yaw_error_rad - self.prev_yaw_error_rad) / dt
        self.filtered_yaw_error_dot_rad = (1.0 - alpha_f) * self.filtered_yaw_error_dot_rad + alpha_f * yaw_error_dot_raw_rad
        self.prev_yaw_error_rad = yaw_error_rad
        yaw_error_dot_rad = self.filtered_yaw_error_dot_rad

        # Pitch
        if self.prev_pitch_error_rad is None: self.prev_pitch_error_rad = pitch_error_rad
        pitch_error_dot_raw_rad = normalize_angle(pitch_error_rad - self.prev_pitch_error_rad) / dt
        self.filtered_pitch_error_dot_rad = (1.0 - alpha_f) * self.filtered_pitch_error_dot_rad + alpha_f * pitch_error_dot_raw_rad
        self.prev_pitch_error_rad = pitch_error_rad
        pitch_error_dot_rad = self.filtered_pitch_error_dot_rad

        # SECOND-ORDER CBF FILTERS (HOCBF) 
        # Enforces hard safety bounds around the AUV during docking.
        
        # Depth HOCBF Filter (Heave uses inverted mapping: e_ddot = +k*u)
        g_sum_z = CBF_GAMMA1_Z + CBF_GAMMA2_Z
        g_prod_z = CBF_GAMMA1_Z * CBF_GAMMA2_Z
        u_heave_cbf_max = ACT_SCALE_Z * (-(g_sum_z * depth_error_dot) + (g_prod_z * (CBF_Z_TOL - depth_error)))
        u_heave_cbf_min = ACT_SCALE_Z * (-(g_sum_z * depth_error_dot) - (g_prod_z * (depth_error + CBF_Z_TOL)))
        if u_heave_cbf_min > u_heave_cbf_max: u_heave_cbf_min = u_heave_cbf_max
        # Clamp nominal control output between the calculated minimum and maximum barrier boundaries
        u_heave_safe = max(u_heave_cbf_min, min(u_heave_nom, u_heave_cbf_max))

        # Sway HOCBF Filter (Sway uses standard mapping: e_ddot = -k*u)
        g_sum_xte = CBF_GAMMA1_XTE + CBF_GAMMA2_XTE
        g_prod_xte = CBF_GAMMA1_XTE * CBF_GAMMA2_XTE
        u_sway_cbf_min = ACT_SCALE_XTE * (g_sum_xte * sway_error_dot - g_prod_xte * (CBF_XTE_TOL - body_sway_error))
        u_sway_cbf_max = ACT_SCALE_XTE * (g_sum_xte * sway_error_dot + g_prod_xte * (body_sway_error + CBF_XTE_TOL))
        if u_sway_cbf_min > u_sway_cbf_max: u_sway_cbf_min = u_sway_cbf_max
        u_sway_safe = max(u_sway_cbf_min, min(u_sway_nom, u_sway_cbf_max))

        # Yaw HOCBF Filter (Yaw uses inverted mapping: e_ddot = +k*u)
        g_sum_yaw = CBF_GAMMA1_YAW + CBF_GAMMA2_YAW
        g_prod_yaw = CBF_GAMMA1_YAW * CBF_GAMMA2_YAW
        u_yaw_cbf_max = ACT_SCALE_YAW * (-(g_sum_yaw * yaw_error_dot_rad) + (g_prod_yaw * (CBF_YAW_TOL_RAD - yaw_error_rad)))
        u_yaw_cbf_min = ACT_SCALE_YAW * (-(g_sum_yaw * yaw_error_dot_rad) - (g_prod_yaw * (yaw_error_rad + CBF_YAW_TOL_RAD)))
        if u_yaw_cbf_min > u_yaw_cbf_max: u_yaw_cbf_min = u_yaw_cbf_max
        u_yaw_safe = max(u_yaw_cbf_min, min(u_yaw_nom, u_yaw_cbf_max))

        # Pitch HOCBF Filter (Pitch uses standard mapping: e_ddot = -k*u)
        g_sum_pitch = CBF_GAMMA1_PITCH + CBF_GAMMA2_PITCH
        g_prod_pitch = CBF_GAMMA1_PITCH * CBF_GAMMA2_PITCH
        u_pitch_cbf_min = ACT_SCALE_PITCH * (g_sum_pitch * pitch_error_dot_rad - g_prod_pitch * (CBF_PITCH_TOL_RAD - pitch_error_rad))
        u_pitch_cbf_max = ACT_SCALE_PITCH * (g_sum_pitch * pitch_error_dot_rad + g_prod_pitch * (pitch_error_rad + CBF_PITCH_TOL_RAD))
        if u_pitch_cbf_min > u_pitch_cbf_max: u_pitch_cbf_min = u_pitch_cbf_max
        u_pitch_safe = max(u_pitch_cbf_min, min(u_pitch_nom, u_pitch_cbf_max))

        # PLATO Implementation
        s_min, s_c, a_max = 0.1, 1.0, 0.2
        n_min, n_max = 3.0, 6.0
        r_d, f_dec, K_ps = 4.0, 7.0, 10.0

        s_t = abs(self.auv_forward_speed)      
        d_t = dist_to_dock                     

        # Calculates dynamic deceleration radius
        R_d_t = r_d + f_dec * (s_t**2) / (2.0 * a_max)
        u_t = 0.0   

        # Yaw‑dependent speed scaling factor
        yaw_abs = abs(yaw_error_deg)
        if yaw_abs > 25.0:
            speed_scale = max(0.2, 1.0 - (yaw_abs - 25.0) / 45.0)   
        else:
            speed_scale = 1.0

        # Execute deceleration profile if inside the critical radius
        if d_t <= R_d_t:
            n_t = n_min + (n_max - n_min) * min(1.0, s_t / s_c)
            safe_d_t = max(d_t, 0.001)
            
            # Map deceleration fraction using log ratios to ensure smooth zero-crossing
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
            u_surge *= speed_scale
        else:
            # Cruise phase logic if outside critical radius
            self.s_d_prev = None
            if abs(yaw_error_deg) > 45.0:
                u_surge = 0.0
            else:
                u_surge = -140.0 * (1.0 - (abs(yaw_error_deg) / 45.0))
            u_surge *= speed_scale

        # Combine control efforts into specific thruster allocation matrix
        u_port = u_surge + u_yaw_safe
        u_stbd = u_surge - u_yaw_safe
        u_heave_b = u_heave_safe + u_pitch_safe
        u_heave_s = u_heave_safe - u_pitch_safe

        self.set_thrust(u_port, u_stbd, u_sway_safe, u_heave_b, u_heave_s)

        self.telemetry.update({
            'phase': 'PLATO Docking', 'dist': dist_to_dock, 'z_err': depth_error,
            'yaw_err': yaw_error_deg, 'xte': body_sway_error, 'speed': self.auv_forward_speed,
            'rd': R_d_t, 'plato_u': u_t
        })

        self.get_logger().info(
            f"[PLATO Phase] Dist2Waypoint: {dist_to_staging:.2f}m | Dist: {dist_to_dock:.2f}m | "
            f"Z-Err: {depth_error:.2f}m | YawErr: {yaw_error_deg:.1f}° | PitchErr: {pitch_error_deg:.1f}° | "
            f"XTE: {body_sway_error:.2f}m | Vis-Align: {self.telemetry['vis_align']} | "
            f"FwdSpeed: {self.auv_forward_speed:.2f}m/s | α: {alpha:.2f}",
            throttle_duration_sec=0.2
        )

class AUVLogger(Node):
    """
    ROS2 Node responsible for thread-safe data collection, buffering, 
    and serialization of telemetry datasets into Parquet format.
    """
    def __init__(self):
        """
        Initializes the logger, creates the data storage directory, 
        sets up the data buffer, and subscribes to auxiliary sensors.
        """
        super().__init__('auv_data_logger')
        self.lock = threading.Lock()

        self.folder_name = "AUVSimData"
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name, exist_ok=True)

        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = os.path.join(self.folder_name, f"AUV_docking_{timestamp_str}.parquet")

        self.data_buffer = []
        self.buffer_limit = 200

        self.sub_gps = self.create_subscription(NavSatFix, '/girona500/gps', self.log_gps, 1)
        self.sub_pre = self.create_subscription(FluidPressure, '/girona500/pressure', self.log_pressure, 1)
        self.sub_dvl = self.create_subscription(TwistWithCovarianceStamped, '/girona500/dvl_twist', self.log_dvl, 1)
        self.sub_acc = self.create_subscription(Imu, '/girona500/accelerometer', self.log_accel, 1)

    def write_line(self, topic, data):
        """
        Appends a timestamped data entry to the in-memory buffer, triggering
        a disk flush if the buffer limit is exceeded.
        
        :param topic: String identifier for the specific sensor or topic.
        :param data: The data payload to log (cast to string internally).
        """
        with self.lock:
            timestamp = self.get_clock().now().to_msg().sec + (self.get_clock().now().to_msg().nanosec / 1e9)
            self.data_buffer.append({'timestamp': timestamp, 'topic': topic, 'data': str(data)})
            if len(self.data_buffer) >= self.buffer_limit:
                self.flush_to_disk_internal()

    def flush_to_disk_internal(self):
        """
        Serializes the internal buffer out to the local parquet file. 
        Note: Must be called while the thread lock is acquired.
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

    def log_val(self, name, val):
        """
        Logs a generic primitive value.
        
        :param name: Identifier name.
        :param val: The value to record.
        """
        self.write_line(name, val)

    def log_odometry(self, msg):
        """
        Logs the 6-DOF odometry pose containing positional translation and quaternion orientation.
        
        :param msg: nav_msgs.msg.Odometry message.
        """
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self.write_line("Odometry_Pose", [p.x, p.y, p.z, o.w, o.x, o.y, o.z])

    def log_imu(self, msg):
        """
        Logs the IMU orientation quaternion.
        
        :param msg: sensor_msgs.msg.Imu message.
        """
        o = msg.orientation
        self.write_line("IMU", [o.x, o.y, o.z, o.w])

    def log_gps(self, msg):
        """
        Logs global positioning coordinates (latitude, longitude).
        
        :param msg: sensor_msgs.msg.NavSatFix message.
        """
        self.write_line("GPS", [msg.latitude, msg.longitude])

    def log_pressure(self, msg):
        """
        Logs external fluid pressure (usually correlating directly to depth).
        
        :param msg: sensor_msgs.msg.FluidPressure message.
        """
        self.write_line("Pressure", msg.fluid_pressure)

    def log_dvl(self, msg):
        """
        Logs DVL (Doppler Velocity Logger) twist characteristics.
        
        :param msg: geometry_msgs.msg.TwistWithCovarianceStamped message.
        """
        v = msg.twist.twist.linear
        cov = list(msg.twist.covariance)
        self.write_line("DVL", [v.x, v.y, v.z, cov])

    def log_accel(self, msg):
        """
        Logs instantaneous linear acceleration metrics.
        
        :param msg: sensor_msgs.msg.Imu message.
        """
        a = msg.linear_acceleration if hasattr(msg, 'linear_acceleration') else msg.accel.linear
        self.write_line("Accelerometer", [a.x, a.y, a.z])

    def log_camera(self, msg):
        """
        Logs metadata attributes (dimensions) of the standard camera feed.
        
        :param msg: sensor_msgs.msg.Image message.
        """
        self.write_line("Main Camera", f"{msg.width}x{msg.height}")

    def log_fls(self, msg):
        """
        Logs metadata attributes (dimensions) of the Forward Looking Sonar (FLS).
        
        :param msg: sensor_msgs.msg.Image message.
        """
        self.write_line("FLS Sonar", f"{msg.width}x{msg.height}")

    def close(self):
        """
        Safely flushes any residual data lingering within the memory buffer before node destruction.
        """
        with self.lock:
            self.flush_to_disk_internal()

# Main Execution and UI
def main(args=None):
    """
    Main entry point for the script. Initializes ROS2 dependencies, creates multithreaded nodes,
    sets up the Matplotlib dashboard interface, and handles lifecycle loops.
    
    :param args: Initialization parameters directly passed via the command line execution.
    """
    rclpy.init(args=args)

    gnc_node = DockingGNC()
    logger_node = gnc_node.data_logger
    vision_node = VisionNode()

    executor = MultiThreadedExecutor()
    executor.add_node(gnc_node)
    executor.add_node(logger_node)
    executor.add_node(vision_node)

    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    
    # Dashboard Setup
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, width_ratios=[2, 1, 1])

    ax_map = fig.add_subplot(gs[0], projection='3d')
    ax_dash = fig.add_subplot(gs[1])
    ax_cbf = fig.add_subplot(gs[2]) 

    # Map Setup
    ax_map.set_title("Live AUV 3D Trajectory")
    ax_map.set_xlabel("X (m)")
    ax_map.set_ylabel("Y (m)")
    ax_map.set_zlabel("Depth (m)")
    ax_map.invert_zaxis()
    ax_map.grid(True)

    path_line, = ax_map.plot([], [], [], 'b-', alpha=0.5, label='AUV Path')
    auv_dot, = ax_map.plot([], [], [], 'bo', markersize=8, label='AUV Position')
    dock_dot, = ax_map.plot([], [], [], 'rX', markersize=10, label='Dock')
    target_line, = ax_map.plot([], [], [], 'g--', alpha=0.8, label='LOS Target Vector')
    ax_map.legend(loc='upper right')

    # Speed / Instrument Setup
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

    # CBF Safety Cross-Section
    ax_cbf.set_title("CBF Safety Corridor")
    ax_cbf.set_aspect('equal')
    ax_cbf.set_xlim(-3.0, 3.0)
    ax_cbf.set_ylim(-3.0, 3.0)
    ax_cbf.set_xlabel("Cross-Track Error (m)")
    ax_cbf.set_ylabel("Depth Error (m)")
    ax_cbf.grid(True, linestyle='--', alpha=0.6)
    
    # Draw the safe boundary lines
    ax_cbf.axhline(1.5, color='red', linestyle='--', linewidth=2, label='Z Limit')
    ax_cbf.axhline(-1.5, color='red', linestyle='--', linewidth=2)
    
    tube_circle = plt.Circle((0, 0), 2.0, color='green', fill=False, linewidth=2, label='Tube Limit')
    ax_cbf.add_patch(tube_circle)
    
    auv_cbf_dot, = ax_cbf.plot([], [], 'bo', markersize=10, label='AUV State')
    ax_cbf.legend(loc='upper right')

    x_history = []
    y_history = []
    z_history = []

    def update_plot(frame):
        """
        The interval callback function for matplotlib FuncAnimation. Accesses AUV state and 
        dock telemetry data continuously to redraw lines, adjust scatter vectors, handle 
        auto-zooming coordinates, and update live strings on the dashboard UI.
        
        :param frame: Standard parameter requirement for matplotlib animations (ignored).
        """
        if gnc_node.dock_pose is None:
            return path_line, auv_dot, dock_dot, target_line, speed_needle, speed_text, telemetry_box, auv_cbf_dot

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

            # Update 3D trajectory plot history
            path_line.set_data(x_history, y_history)
            path_line.set_3d_properties(z_history)

            # Update live AUV and Dock marker positions
            auv_dot.set_data([curr_x], [curr_y])
            auv_dot.set_3d_properties([curr_z])

            dock_dot.set_data([dock_x], [dock_y])
            dock_dot.set_3d_properties([dock_z])

            # Calculate and draw target lookahead vector projection
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

            # Auto-scale axes dynamically to maintain focus on the AUV and Dock
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

            # Update dashboard instrument logic
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
                f"PLATO Accel U: {t['plato_u']:6.3f}  \n"
                f"Vis Align    : {t.get('vis_align', 'WAITING')} \n"
                f"Blend α      : {t.get('alpha', 0.0):.2f}"
            )
            telemetry_box.set_text(dash_str)
            
            # Update the CBF Safety Monitor dot against cross-track vs depth axes
            auv_cbf_dot.set_data([t['xte']], [t['z_err']])

            # Turn CBF dot red if boundaries are exceeded
            if abs(t['xte']) > 2.0 or abs(t['z_err']) > 1.5:
                auv_cbf_dot.set_color('red')
            else:
                auv_cbf_dot.set_color('blue')

        return path_line, auv_dot, dock_dot, target_line, speed_needle, speed_text, telemetry_box, auv_cbf_dot

    ani = animation.FuncAnimation(fig, update_plot, interval=100, cache_frame_data=False)

    try:
        plt.show() 
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down plotting and stopping AUV Thrusters...")
        gnc_node.set_thrust(0.0, 0.0, 0.0, 0.0, 0.0)
        logger_node.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()