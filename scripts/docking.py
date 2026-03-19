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

# Sensor Noise Configuration
# Standard Deviation (sigma) for Gaussian Noise
NOISE_POS_STD   = 0.15   # +/- 15cm jitter (DVL/USBL noise)
NOISE_DEPTH_STD = 0.05   # +/- 5cm jitter
NOISE_YAW_STD   = 0.05   # +/- ~3 degrees jitter (Compass/IMU noise)
NOISE_DOCK_STD  = 0.2    # +/- 20cm uncertainty in detecting the dock

# GNC Parameters
KP_HEADING      = -4.5       # Yaw proportional gain (Steering)
KI_HEADING      = -0.1       # Yaw integral gain
KP_SWAY         = 50.0       # Sway gain (Cross-track error correction)
INTEGRAL_LIMIT  = 15.0

# Depth Control Parameters
KP_DEPTH        = 80.0       # High gain to overcome positive buoyancy
KI_DEPTH        = 5.0        
DEPTH_I_LIMIT   = 20.0       

# Pure Pursuit Parameters
LOOKAHEAD_DIST  = 6.0        
DOCKING_TOLERANCE = 0.5      

LIGHT_COORDS = [
    [0.0, 4.53528, -2.58534],        # Light 1
    [2.585345, 4.53528, 0.0],        # Light 3
    [0.0, 4.53528, 2.585345],        # Light 5
    [-2.585345, 4.53528, 0.0]        # Light 7
]

def get_yaw(q):
    """Converts Quaternion to Yaw angle."""
    t3 = +2.0 * (q.w * q.z + q.x * q.y)
    t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(t3, t4)

def normalize_angle(angle):
    """Keeps angle between -pi and pi."""
    while angle > math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle

def quaternion_matrix(q):
    """Returns a 3x3 rotation matrix from a quaternion (w, x, y, z)."""
    w, x, y, z = q.w, q.x, q.y, q.z
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])

class DockingGNC(Node):
    def __init__(self):
        super().__init__('auv_docking_gnc')

        # State Variables
        self.dock_pose = None 
        self.auv_x = 0.0
        self.auv_y = 0.0
        self.auv_z = 0.0 
        self.auv_yaw = 0.0
        self.auv_forward_speed = 0.0
        
        self.light_center_local = np.mean(LIGHT_COORDS, axis=0) 
        
        self.integral_error = 0.0
        self.depth_integral = 0.0 
        self.prev_time = self.get_clock().now()
        
        # Thrusters (QoS set to 1 for faster, unreliable delivery if network is congested)
        self.pub_port = self.create_publisher(Float64, '/girona500/ThrusterSurgePort/setpoint', 1)
        self.pub_stbd = self.create_publisher(Float64, '/girona500/ThrusterSurgeStarboard/setpoint', 1)
        self.pub_sway = self.create_publisher(Float64, '/girona500/ThrusterSway/setpoint', 1)
        self.pub_heave_b = self.create_publisher(Float64, '/girona500/ThrusterHeaveBow/setpoint', 1)
        self.pub_heave_s = self.create_publisher(Float64, '/girona500/ThrusterHeaveStern/setpoint', 1)

        # Sensors
        self.create_subscription(Odometry, '/girona500/dynamics/odometry', self.auv_cb, 1)
        self.create_subscription(Odometry, '/dock/position', self.dock_cb, 1)
        
        # Initialize Logger
        self.data_logger = AUVLogger()
        self.add_logger_subscribers()

        self.timer = self.create_timer(0.02, self.control_loop)
        self.get_logger().info("Docking Node Started.")

    def add_logger_subscribers(self):
        # Using lambda to hook directly into the logger
        self.create_subscription(Float64, '/girona500/ThrusterSurgePort/setpoint', lambda m: self.data_logger.log_val('ThrusterSurgePort', m.data), 1)
        self.create_subscription(Float64, '/girona500/ThrusterSurgeStarboard/setpoint', lambda m: self.data_logger.log_val('ThrusterSurgeStarboard', m.data), 1)
        self.create_subscription(Float64, '/girona500/ThrusterSway', lambda m: self.data_logger.log_val('ThrusterSway', m.data), 1)
        self.create_subscription(Float64, '/girona500/ThrusterHeaveBow/setpoint', lambda m: self.data_logger.log_val('ThrusterHeaveBow', m.data), 1)
        self.create_subscription(Float64, '/girona500/ThrusterHeaveStern/setpoint', lambda m: self.data_logger.log_val('ThrusterHeaveStern', m.data), 1)
        self.create_subscription(Odometry, '/girona500/dynamics/odometry', self.data_logger.log_odometry, 1)
        self.create_subscription(Imu, '/girona500/imu', self.data_logger.log_imu, 1)
        self.create_subscription(Image, '/girona500/camera/image_raw', self.data_logger.log_camera, 1)
        self.create_subscription(Image, '/girona500/fls/image_raw', self.data_logger.log_fls, 1)


    def dock_cb(self, msg):
        self.dock_pose = msg.pose.pose

        # Add noise to the perceived dock position
        if self.dock_pose:
            self.dock_pose.position.x += np.random.normal(0, NOISE_DOCK_STD)
            self.dock_pose.position.y += np.random.normal(0, NOISE_DOCK_STD)
            self.dock_pose.position.z += np.random.normal(0, NOISE_DOCK_STD)

    def auv_cb(self, msg):
        # Add noise to the AUV Odometry readings
        # Calculate the clean values first, then add noise.
        
        clean_x = msg.pose.pose.position.x
        clean_y = msg.pose.pose.position.y
        clean_z = msg.pose.pose.position.z
        clean_yaw = get_yaw(msg.pose.pose.orientation)

        # Apply Gaussian noise
        self.auv_x = clean_x + np.random.normal(0, NOISE_POS_STD)
        self.auv_y = clean_y + np.random.normal(0, NOISE_POS_STD)
        self.auv_z = clean_z + np.random.normal(0, NOISE_DEPTH_STD)
        
        # Apply Noise to Yaw (and re-normalize to keep it valid)
        noisy_yaw = clean_yaw + np.random.normal(0, NOISE_YAW_STD)
        self.auv_yaw = normalize_angle(noisy_yaw)

        # Extract forward speed directly from Odometry twist (no noise added here)
        self.auv_forward_speed = msg.twist.twist.linear.x

    # def auv_cb(self, msg):
    #     self.auv_x = msg.pose.pose.position.x
    #     self.auv_y = msg.pose.pose.position.y
    #     self.auv_z = msg.pose.pose.position.z 
    #     self.auv_yaw = get_yaw(msg.pose.pose.orientation)

    def set_thrust(self, port, stbd, sway, heave):
        port = max(min(port, 200.0), -200.0)
        stbd = max(min(stbd, 200.0), -200.0)
        sway = max(min(sway, 200.0), -200.0)
        heave = max(min(heave, 200.0), -200.0)

        self.pub_port.publish(Float64(data=float(port)))
        self.pub_stbd.publish(Float64(data=float(stbd)))
        self.pub_sway.publish(Float64(data=float(sway)))
        self.pub_heave_b.publish(Float64(data=float(heave)))
        self.pub_heave_s.publish(Float64(data=float(heave)))

    def control_loop(self):
        if self.dock_pose is None:
            return

        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds / 1e9
        if dt <= 0: return
        self.prev_time = current_time
        
        # Geometry Logic
        P_tail_2d = np.array([self.dock_pose.position.x, self.dock_pose.position.y])
        dock_z_target = self.dock_pose.position.z
        
        R_dock = quaternion_matrix(self.dock_pose.orientation)
        offset_rotated = R_dock.dot(self.light_center_local)
        
        P_mouth_2d = P_tail_2d + offset_rotated[:2] 
        
        vec_path = P_tail_2d - P_mouth_2d
        len_path = np.linalg.norm(vec_path)
        
        if len_path < 0.001: return 
        
        u_path = vec_path / len_path 
        P_auv_2d = np.array([self.auv_x, self.auv_y])
        
        vec_MA = P_auv_2d - P_mouth_2d
        s_curr = np.dot(vec_MA, u_path)
        P_closest = P_mouth_2d + (s_curr * u_path)
        P_goal = P_closest + (LOOKAHEAD_DIST * u_path)
        
        # Controllers

        # Depth Control 
        depth_error = dock_z_target - self.auv_z
        
        self.depth_integral += depth_error * dt
        self.depth_integral = max(min(self.depth_integral, DEPTH_I_LIMIT), -DEPTH_I_LIMIT)
        
        # INVERTED: We multiply by -1.0 to ensure positive error results in negative thrust (Down)
        u_heave = -1.0 * ((KP_DEPTH * depth_error) + (KI_DEPTH * self.depth_integral))

        # Distance to reach dock check
        P_tail_3d = np.array([self.dock_pose.position.x, self.dock_pose.position.y, self.dock_pose.position.z])
        P_auv_3d = np.array([self.auv_x, self.auv_y, self.auv_z])
        
        dist_to_dock = np.linalg.norm(P_tail_3d - P_auv_3d)
        
        if dist_to_dock < DOCKING_TOLERANCE:
            # Log less frequently when stopped to save space.
            self.get_logger().info("DOCKED! Stopping Thrusters.")
            self.set_thrust(0.0, 0.0, 0.0, 0.0) 
            return

        # Heading Control (LOS)
        vec_to_goal = P_goal - P_auv_2d
        desired_yaw = math.atan2(vec_to_goal[1], vec_to_goal[0])
        
        yaw_error = normalize_angle(desired_yaw - self.auv_yaw)
        yaw_error_deg = math.degrees(yaw_error)
        
        self.integral_error += yaw_error * dt
        self.integral_error = max(min(self.integral_error, INTEGRAL_LIMIT), -INTEGRAL_LIMIT)
        
        u_yaw = (KP_HEADING * yaw_error_deg) + (KI_HEADING * self.integral_error)
        
        # Sway Control (Cross Track Error)
        vec_err_world = P_closest - P_auv_2d
        c = math.cos(self.auv_yaw)
        s = math.sin(self.auv_yaw)
        body_sway_error = (-s * vec_err_world[0]) + (c * vec_err_world[1])
        u_sway = KP_SWAY * body_sway_error

        # Surge Control
        if abs(yaw_error_deg) > 45.0:
            u_surge = 0.0
        else:
            # Dynamically scale surge: faster when aligned, slower when turning
            # Max base surge is -140.0, leaving 60.0 units of thrust for steering
            u_surge = -140.0 * (1.0 - (abs(yaw_error_deg) / 45.0))

        # Actuation mixing
        u_port = u_surge + u_yaw
        u_stbd = u_surge - u_yaw

        self.set_thrust(u_port, u_stbd, u_sway, u_heave)
        
        # Note: Printing to console at 50Hz might lag the terminal. 
        # We might throttle this print if it's too fast.
        self.get_logger().info(
            f"Dist: {dist_to_dock:.2f}m | Z-Err: {depth_error:.2f}m | YawErr: {yaw_error_deg:.1f}° | XTE: {body_sway_error:.2f}m | FwdSpeed: {self.auv_forward_speed:.2f}m/s",
            throttle_duration_sec=0.2 
        )


# class AUVLogger(Node):
#     def __init__(self):
#         super().__init__('auv_data_logger')
        
#         self.folder_name = "AUVSimData"
#         if not os.path.exists(self.folder_name):
#             try:
#                 os.makedirs(self.folder_name)
#                 self.get_logger().info(f"Created log directory: {self.folder_name}")
#             except OSError as e:
#                 self.get_logger().error(f"Failed to create directory {self.folder_name}: {e}")
#                 self.folder_name = "." 

#         now = datetime.now()
#         timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        
#         self.filename = os.path.join(self.folder_name, f"AUV_docking_{timestamp_str}.parquet")
        
#         # Parquet Optimization: Use a buffer to avoid frequent disk I/O
#         self.data_buffer = []
#         self.buffer_limit = 100 

#         # Subscriptions for logging set to QoS 1 for lower latency/overhead
#         self.sub_gps = self.create_subscription(NavSatFix, '/girona500/gps', self.log_gps, 1)
#         self.sub_pre = self.create_subscription(FluidPressure, '/girona500/pressure', self.log_pressure, 1)
#         self.sub_dvl = self.create_subscription(TwistWithCovarianceStamped, '/girona500/dvl_twist', self.log_dvl, 1) 
#         self.sub_acc = self.create_subscription(Imu, '/girona500/accelerometer', self.log_accel, 1)

#     def write_line(self, topic, data):
#         timestamp = self.get_clock().now().to_msg().sec + (self.get_clock().now().to_msg().nanosec / 1e9)
        
#         # Convert list data to strings to ensure consistent DataFrame column shapes
#         # and prevent the "Length mismatch" index error.
#         log_entry = {
#             'timestamp': timestamp,
#             'topic': topic,
#             'data': str(data) 
#         }
        
#         self.data_buffer.append(log_entry)

#         if len(self.data_buffer) >= self.buffer_limit:
#             self.flush_to_disk()

#     def flush_to_disk(self):
#         if not self.data_buffer:

#             return
        
#         try:
#             # Create DF and explicitly reset index to avoid length mismatch errors
#             df = pd.DataFrame(self.data_buffer)
#             df.reset_index(drop=True, inplace=True)
            
#             # pyarrow is generally more stable for appending than fastparquet
#             file_exists = os.path.isfile(self.filename)
#             if not file_exists:
#                 df.to_parquet(self.filename, engine='pyarrow', index=False)
#             else:
#                 # Append mode for pyarrow requires using the ParquetWriter or 
#                 # simply reading/concatenating if the file is small.
#                 # For AUV missions, reading/appending is safer to prevent file corruption.
#                 existing_df = pd.read_parquet(self.filename)
#                 combined_df = pd.concat([existing_df, df], ignore_index=True)
#                 combined_df.to_parquet(self.filename, engine='pyarrow', index=False)
            
#             self.data_buffer = []
#         except Exception as e:
#             self.get_logger().error(f"Failed to flush logs: {e}")

#     def write_header(self):
#         # Parquet files handle metadata automatically; headers aren't needed.
#         pass

#     def log_val(self, name, val):
#         self.write_line(name, val)

#     def log_odometry(self, msg):
#         # Recording Full Pose: [Position XYZ, Orientation WXYZ]
#         p = msg.pose.pose.position
#         o = msg.pose.pose.orientation
#         self.write_line("Odometry_Pose", [p.x, p.y, p.z, o.w, o.x, o.y, o.z])

#     def log_imu(self, msg):
#         o = msg.orientation
#         self.write_line("IMU", [o.x, o.y, o.z, o.w])

#     def log_gps(self, msg):
#         self.write_line("GPS", [msg.latitude, msg.longitude])

#     def log_pressure(self, msg):
#         self.write_line("Pressure", msg.fluid_pressure)

#     def log_dvl(self, msg):
#         # Recording DVL Velocity and Covariance
#         v = msg.twist.twist.linear
#         cov = msg.twist.covariance
#         self.write_line("DVL", [v.x, v.y, v.z, list(cov)])

#     def log_accel(self, msg):
#         if hasattr(msg, 'linear_acceleration'):
#             a = msg.linear_acceleration
#             self.write_line("Accelerometer", [a.x, a.y, a.z])
#         elif hasattr(msg, 'accel'):
#              a = msg.accel.linear
#              self.write_line("Accelerometer", [a.x, a.y, a.z])
    
#     def log_camera(self, msg):
#         self.write_line("Main Camera", f"{msg.width}x{msg.height}")

#     def log_fls(self, msg):
#         self.write_line("FLS Sonar", f"{msg.width}x{msg.height}")

#     def close(self):
#         self.flush_to_disk()

class AUVLogger(Node):
    def __init__(self):
        super().__init__('auv_data_logger')
        
        # Thread safety lock
        self.lock = threading.Lock()
        
        self.folder_name = "AUVSimData"
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name, exist_ok=True)

        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = os.path.join(self.folder_name, f"AUV_docking_{timestamp_str}.parquet")
        
        self.data_buffer = []
        self.buffer_limit = 200 # Increased limit to reduce disk write frequency

        # Subscriptions
        self.sub_gps = self.create_subscription(NavSatFix, '/girona500/gps', self.log_gps, 1)
        self.sub_pre = self.create_subscription(FluidPressure, '/girona500/pressure', self.log_pressure, 1)
        self.sub_dvl = self.create_subscription(TwistWithCovarianceStamped, '/girona500/dvl_twist', self.log_dvl, 1) 
        self.sub_acc = self.create_subscription(Imu, '/girona500/accelerometer', self.log_accel, 1)

    def write_line(self, topic, data):
        # Use a lock to prevent simultaneous access from different ROS callback threads
        with self.lock:
            timestamp = self.get_clock().now().to_msg().sec + (self.get_clock().now().to_msg().nanosec / 1e9)
            
            # Save data as string to ensure uniform column shape in the Parquet file
            self.data_buffer.append({
                'timestamp': timestamp,
                'topic': topic,
                'data': str(data)
            })

            if len(self.data_buffer) >= self.buffer_limit:
                self.flush_to_disk_internal()

    def flush_to_disk_internal(self):
        """Internal helper called while lock is held."""
        if not self.data_buffer:
            return
        
        try:
            df = pd.DataFrame(self.data_buffer)
            
            # Using pyarrow with index=False is the most robust for AUV sensor data
            if not os.path.isfile(self.filename):
                df.to_parquet(self.filename, engine='pyarrow', index=False)
            else:
                # Append by reading existing and concatenating
                # This prevents the 'invalid TType' thrift corruption error
                existing_df = pd.read_parquet(self.filename, engine='pyarrow')
                pd.concat([existing_df, df], ignore_index=True).to_parquet(self.filename, engine='pyarrow', index=False)
            
            self.data_buffer = []
        except Exception as e:
            self.get_logger().error(f"Log Flush Error: {e}")

    def log_val(self, name, val):
        self.write_line(name, val)

    def log_odometry(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self.write_line("Odometry_Pose", [p.x, p.y, p.z, o.w, o.x, o.y, o.z])

    def log_imu(self, msg):
        o = msg.orientation
        self.write_line("IMU", [o.x, o.y, o.z, o.w])

    def log_gps(self, msg):
        self.write_line("GPS", [msg.latitude, msg.longitude])

    def log_pressure(self, msg):
        self.write_line("Pressure", msg.fluid_pressure)

    def log_dvl(self, msg):
        v = msg.twist.twist.linear
        cov = list(msg.twist.covariance)
        self.write_line("DVL", [v.x, v.y, v.z, cov])

    def log_accel(self, msg):
        a = msg.linear_acceleration if hasattr(msg, 'linear_acceleration') else msg.accel.linear
        self.write_line("Accelerometer", [a.x, a.y, a.z])
    
    def log_camera(self, msg):
        self.write_line("Main Camera", f"{msg.width}x{msg.height}")

    def log_fls(self, msg):
        self.write_line("FLS Sonar", f"{msg.width}x{msg.height}")

    def close(self):
        with self.lock:
            self.flush_to_disk_internal()
        
def main(args=None):
    rclpy.init(args=args)
    
    gnc_node = DockingGNC()
    logger_node = gnc_node.data_logger
    
    executor = MultiThreadedExecutor()
    executor.add_node(gnc_node)
    executor.add_node(logger_node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        gnc_node.set_thrust(0.0, 0.0, 0.0, 0.0)
        logger_node.close()
        gnc_node.destroy_node()
        logger_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()